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

    action = root.get_group("action")
    obs = root.get_group("obs")

    times = obs.get_array("timestamp")[starts[index] : ends[index]]
    times = np.asarray(times, np.float64)
    times -= times[0]

    dt = times[1:] - times[:-1]
    hz = 1 / dt
    hz = np.concatenate([np.zeros(1, dtype=hz.dtype), hz])

    rr.init(f"{index}", spawn=not save)

    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial2DView(name="Wrist Camera", origin="obs/wrist_img"),
                rrb.Spatial2DView(name="Front Camera", origin="obs/front_img"),
            ),
            rrb.Horizontal(
                rrb.TimeSeriesView(
                    name="Joint Positions",
                    origin="obs/qpos",
                ),
                rrb.TimeSeriesView(name="Joint Velocities", origin="obs/qvel"),
                rrb.TimeSeriesView(name="Action Joint Positions", origin="action/qpos"),
            ),
            rrb.TimeSeriesView(
                name="Hz",
                origin="hz",
            ),
            rrb.TimeSeriesView(
                name="dt",
                origin="dt",
            ),
            row_shares=[3.5, 2, 0.5, 0.5],
        ),
        rrb.BlueprintPanel(state="collapsed"),
        rrb.SelectionPanel(state="collapsed"),
        rrb.TimePanel(state="collapsed", play_state="paused"),
    )

    rr.send_blueprint(blueprint)

    colors = [(np.random.rand(3) * 255).tolist() for _ in range(7)]

    for i in range(7):
        rr.log(f"obs/qpos/{i}", rr.SeriesLines(colors=colors[i]), static=True)
        rr.log(f"obs/qvel/{i}", rr.SeriesLines(colors=colors[i]), static=True)
        rr.log(f"action/qpos/{i}", rr.SeriesLines(colors=colors[i]), static=True)

    for i, t in enumerate(times.tolist()):
        if i % 9 != 0:
            continue

        rr.set_time("time", duration=t)

        rr.log("hz", rr.Scalars(hz[i]))
        rr.log("dt", rr.Scalars(obs.get_array("dt")[i]))

        for j in range(7):
            rr.log(f"obs/qpos/{j}", rr.Scalars(scalars=obs.get_array("qpos")[i, j]))

            rr.log(f"obs/qvel/{j}", rr.Scalars(obs.get_array("qvel")[i, j]))

        img = obs.get_array("wrist_img")[i]
        img = np.asarray(img, dtype=np.uint8)
        rr.log("obs/wrist_img", rr.Image(img))

        img = obs.get_array("front_img")[i]
        img = np.asarray(img, dtype=np.uint8)
        rr.log("obs/front_img", rr.Image(img))

        for j in range(7):
            rr.log(f"action/qpos/{j}", rr.Scalars(action.get_array("qpos")[i, j]))

        time.sleep(1e-6)

    if save:
        rr.save(f"{index}.rrd")


if __name__ == "__main__":
    main()
