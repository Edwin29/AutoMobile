"""Known planar geometry, explicit boundary IDs and route/landmark generation."""
from pathlib import Path
from dataclasses import dataclass
import json
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.ndimage import map_coordinates
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize

from .geometry import footprint, to_world, path_geometry
from .camera import sha256_file

HERE = Path(__file__).parent
ARMS = {"N": np.array([0., 1.]), "E": np.array([1., 0.]), "S": np.array([0., -1.]), "W": np.array([-1., 0.])}


def box_sdf(points, box, radius):
    p = np.asarray(points)
    x0, x1, y0, y1 = box
    q = np.abs(p-np.array([(x0+x1)/2, (y0+y1)/2]))-np.array([(x1-x0)/2-radius, (y1-y0)/2-radius])
    return np.linalg.norm(np.maximum(q, 0), axis=-1)+np.minimum(np.max(q, axis=-1), 0)-radius


@dataclass
class Route:
    id: str
    points: np.ndarray
    closed: bool = False
    junction: str | None = None
    entry: str | None = None
    exit: str | None = None
    clearance: float | None = None
    generation: dict | None = None

    def __post_init__(self):
        self.s, self.heading, self.curvature = path_geometry(self.points)

    def nearest(self, position):
        return int(np.argmin(np.linalg.norm(self.points-position, axis=1)))

    def sample(self, arc):
        values = np.asarray(arc)
        values = values % self.s[-1] if self.closed else np.clip(values, 0, self.s[-1])
        return np.c_[np.interp(values, self.s, self.points[:, 0]), np.interp(values, self.s, self.points[:, 1])]


class TrackMap:
    def __init__(self, path=HERE/"config/map.json"):
        self.path = Path(path)
        self.data = json.loads(self.path.read_text(encoding="utf-8"))
        self.id = sha256_file(path)
        self.boundary_ids = ["outer"]+[f"island_{i}" for i in range(len(self.data["blocks"]))]
        self.landmarks = []
        self._grid = None
        self._junction_cache = {}
        for i, (x,y,d) in enumerate(self.data["crosswalks"]):
            self.landmarks.append(dict(id=f"cross_{i:02d}", kind="crosswalk", xy=[x,y],
                                       approach=d, central=i<4, sigma_cm=self.data["position_uncertainty_cm"]))
        for kind, key in (("dashline", "pink_lines"), ("red_line", "red_lines")):
            for i, ends in enumerate(self.data[key]):
                self.landmarks.append(dict(id=f"{kind}_{i:02d}", kind=kind, xy=np.mean(ends,axis=0).tolist(),
                                           endpoints=ends, sigma_cm=self.data["position_uncertainty_cm"]))
        disks = [[x,y] for x in self.data["blue_x"] for y in self.data["blue_y"]]+self.data["blue_extra"]
        for i, xy in enumerate(disks):
            self.landmarks.append(dict(id=f"blue_{i:02d}", kind="blue_circle_candidate", xy=xy,
                                       radius_cm=self.data["blue_radius_cm"], sigma_cm=self.data["position_uncertainty_cm"]))

    def distances(self, points):
        return np.stack([-box_sdf(points,self.data["outer"],self.data["outer_radius_cm"])] +
                        [box_sdf(points,b,self.data["block_radius_cm"]) for b in self.data["blocks"]], axis=-1)

    def clearance(self, points):
        return self.distances(points).min(axis=-1)

    def clearance_fast(self, points):
        """Conservative interpolated SDF lower bound for repeated local planning.

        Signed distance is 1-Lipschitz. Subtracting the cell diagonal bounds
        interpolation error; exact geometry remains the independent evaluator.
        """
        step=.25;origin=-130.
        if self._grid is None:
            yy,xx=np.mgrid[:1041,:1041]
            self._grid=self.clearance(np.stack((xx*step+origin,yy*step+origin),axis=-1))
        p=np.asarray(points); flat=p.reshape(-1,2)
        coordinates=((flat-origin)/step)[:,::-1].T
        return (map_coordinates(self._grid,coordinates,order=1,mode="constant",cval=-100.,prefilter=False)
                -np.sqrt(2)*step).reshape(p.shape[:-1])

    def boundary_residual(self, points, ids):
        d = self.distances(points)
        return d[np.arange(len(d)), ids]

    def route_clearance(self, route, vehicle):
        rim = footprint(vehicle)
        # All poses, with both perimeter and interior, evaluated in vectorized batches.
        h = route.heading
        c,s = np.cos(h),np.sin(h)
        world = np.empty((len(h),len(rim),2))
        world[:,:,0] = route.points[:,None,0]+c[:,None]*rim[None,:,0]+s[:,None]*rim[None,:,1]
        world[:,:,1] = route.points[:,None,1]-s[:,None]*rim[None,:,0]+c[:,None]*rim[None,:,1]
        return self.clearance(world).min(axis=1)

    def outer_route(self, direction="cw", optimize=True):
        # Archive route contains no driving code and is only a starting geometric path.
        with np.load(HERE.parent/"map_localization/outer_driver/track.npz",allow_pickle=False) as z:
            p = z["route"][::30].copy()
        p = np.vstack((p,p[0]))
        s = np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
        spline = make_interp_spline(s,p,k=3,bc_type="periodic")
        dense = spline(np.linspace(0,s[-1],int(s[-1]/.5)+1))
        if direction == "ccw":
            dense = dense[::-1].copy()
        elif direction != "cw":
            raise ValueError("Unknown direction")
        route=Route(f"outer_{direction}",dense,True)
        if optimize:
            from .settings import load_settings
            vehicle=load_settings()[0]["vehicle"]
            # Offline lateral clearance search, then periodic smoothing. Re-evaluate
            # the REAL resulting tangents; do not reuse old footprint orientations.
            right=np.c_[np.cos(route.heading),-np.sin(route.heading)]
            offsets=np.arange(-3.,3.01,.25)
            margins=[]
            for d in offsets:
                trial=Route("trial",dense+right*d,True)
                margins.append(self.route_clearance(trial,vehicle))
            best=offsets[np.argmax(np.asarray(margins),axis=0)]
            best=gaussian_filter1d(best,12,mode="wrap")
            best[-1]=best[0]
            choices=[route]
            for scale in (.5,.75,1.):
                trial=Route(route.id,dense+right*best[:,None]*scale,True)
                trial.clearance=float(self.route_clearance(trial,vehicle).min());choices.append(trial)
            route.clearance=float(self.route_clearance(route,vehicle).min())
            route=max(choices,key=lambda r:r.clearance)
        return route

    def turn_name(self, entry, exit):
        incoming, outgoing = -ARMS[entry], ARMS[exit]
        cross = incoming[0]*outgoing[1]-incoming[1]*outgoing[0]
        return "straight" if np.dot(incoming,outgoing) > .9 else ("left" if cross > 0 else "right")

    def junction_routes(self, junction, entry, vehicle):
        cache_key=(junction,entry,tuple(sorted(vehicle.items())))
        if cache_key in self._junction_cache:
            import copy
            return copy.deepcopy(self._junction_cache[cache_key])
        node = self.data["junctions"][junction]
        if entry not in node["arms"]:
            raise ValueError("Invalid entry arm")
        center = np.asarray(node["xy"],float)
        routes = {}
        for exit in node["arms"]:
            if exit == entry:
                continue
            a,b = center+35*ARMS[entry],center+35*ARMS[exit]
            def make(handles):
                controls = np.array([a,a-handles[0]*ARMS[entry],b-handles[1]*ARMS[exit],b])
                t = np.linspace(0,1,181)[:,None]
                p = (1-t)**3*controls[0]+3*(1-t)**2*t*controls[1]+3*(1-t)*t**2*controls[2]+t**3*controls[3]
                route = Route(f"{junction}_{entry}_{exit}",p,False,junction,entry,exit)
                route.clearance = float(self.route_clearance(route,vehicle).min())
                route.generation={"family":"cubic_bezier","handles_cm":list(map(float,handles)),"optimization":"bounded_two_handles"}
                return route
            def cost(handles):
                route=make(handles)
                excess=max(0.,float(np.max(np.abs(route.curvature)))-.14)
                return -route.clearance+100*excess
            starts=[(r,r) for r in (15.,22.,30.,38.,46.)]
            initial=min(starts,key=cost)
            fit=minimize(cost,initial,method="Powell",bounds=[(8.,55.),(8.,55.)],options={"maxiter":12,"maxfev":80,"xtol":.1,"ftol":.01})
            best=make(fit.x if cost(fit.x)<cost(initial) else initial)
            routes[self.turn_name(entry,exit)] = best
        self._junction_cache[cache_key]=routes
        import copy
        return copy.deepcopy(routes)

    def margin_table(self, route, vehicle, lateral=(-1.,0.,1.), yaw=(-.08,0.,.08), stride=10):
        rows = []
        rim = footprint(vehicle)
        for i in range(0,len(route.points),stride):
            h=route.heading[i]
            for d in lateral:
                for psi in yaw:
                    pos=route.points[i]+d*np.array([np.cos(h),-np.sin(h)])
                    value=float(self.clearance(to_world(rim,[*pos,h+psi])).min())
                    rows.append([float(route.s[i]),d,psi,value])
        return rows
