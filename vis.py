import time

import click
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import zarr


@click.command()
@click.option("--dataset", default="data")
@click.option("--index", default=0)
@click.option("--save", is_flag=True)
def main(dataset, index, save):
    root = zarr.group(dataset)

    ends = root.get_array("episode_ends")[:]
    ends = np.asarray(ends, dtype=np.uint64)

    starts = np.concatenate([np.zeros(1, dtype=np.uint64), ends[:-1]])

    rr.init(f"{index}", spawn=not save)

    action_group = root.get_group("action")

    action = {}

    for name in action_group.array_keys():
        z = action_group.get_array(name)[starts[index] : ends[index]]
        action[name] = np.asarray(z, dtype=np.float64)

    obs_group = root.get_group("obs")

    obs = {}

    for name in obs_group.array_keys():
        z = obs_group.get_array(name)[starts[index] : ends[index]]

        if "img" in name:
            obs[name] = np.asarray(z, dtype=np.uint8)
        else:
            obs[name] = np.asarray(z, dtype=np.float64)

    times = obs["timestamp"]
    times = np.asarray(times, np.float64)
    times -= times[0]

    for i, t in enumerate(times.tolist()):
        for j in range(7):
            rr.set_time("time", duration=t)

            rr.log(f"obs/qpos/{j}", rr.Scalars(scalars=obs["qpos"][i, j]))

            rr.log(f"obs/qvel/{j}", rr.Scalars(obs["qvel"][i, j]))

        rr.log("obs/wrist_img", rr.Image(obs["wrist_img"][i]))

        rr.log("obs/front_img", rr.Image(obs["front_img"][i]))

        for j in range(7):
            rr.log(f"action/qpos/{j}", rr.Scalars(action["qpos"][i, j]))

        time.sleep(1e-3)

    blueprint = rrb.Grid(
        rrb.TimeSeriesView(
            name="Joint Positions",
            origin="obs/qpos",
        ),
        rrb.TimeSeriesView(name="Joint Velocities", origin="obs/qvel"),
        rrb.TimeSeriesView(name="Action Joint Positions", origin="action/qpos"),
        rrb.Spatial2DView(name="Wrist Camera", origin="obs/wrist_img"),
        rrb.Spatial2DView(name="Front Camera", origin="obs/front_img"),
    )

    rr.send_blueprint(blueprint)

    if save:
        rr.save(f"{index}.rrd")


if __name__ == "__main__":
    main()
