from argparse import ArgumentParser
from dataclasses import dataclass
from imgui_bundle import immapp
from imgui_bundle._imgui_bundle import imgui, hello_imgui
from pyvista_imgui import ImguiPlotter
from scipy.spatial.transform import Rotation
from utils.env_containers import EnvContainer
from utils.ppo import MyPPO
from utils.tree_utils import tree_stack
from utils.vecenv import MyVecEnv
from viz.visual_data import XMLVisualDataContainer

import matplotlib
import numpy as np
import os
import pyvista as pv

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


@dataclass
class VisualArray:
    meshes: np.ndarray
    actors: np.ndarray


class AppState:
    def __init__(
        self,
        scene_visuals: VisualArray,
        trail_visuals: VisualArray,
        train_vecenv: MyVecEnv,
        eval_vecenv: MyVecEnv,
        ppo: MyPPO,
    ):
        self.scene_meshes = scene_visuals.meshes
        self.scene_actors = scene_visuals.actors
        self.trail_meshes = trail_visuals.meshes
        self.trail_actors = trail_visuals.actors
        self.train_vecenv = train_vecenv
        self.eval_vecenv = eval_vecenv
        self.ppo = ppo
        _, self.ppo_callback = self.ppo._setup_learn(int(1e6), None)

        # GUI state parameters, in alphabetical order
        self.color_code = 0
        self.deterministic = False
        self.eval_obs = None
        self.eval_rewards = np.zeros((1, 1), dtype=float)
        self.first_time = True
        self.iter_i = 0
        self.iterating = False
        self.n_epochs = 1
        self.n_iters = 10
        self.play_mode = False
        self.pose_idx = 0
        self.rollout_length = self.ppo.n_steps
        self.show_axes = False
        self.show_guides = True
        self.show_trails = False
        self.traj_frame = 0
        self.traj_idx = 0
        self.trajectory_t = 0
        self.trajectory_x = None


def setup_and_run_gui(pl: ImguiPlotter, app_state: AppState):
    runner_params = hello_imgui.RunnerParams()
    runner_params.app_window_params.window_title = "Viewer"
    runner_params.app_window_params.window_geometry.size = (1280, 720)

    def gui():
        hello_imgui.apply_theme(hello_imgui.ImGuiTheme_.imgui_colors_dark)

        viewport_size = imgui.get_window_viewport().size

        # PyVista portion
        imgui.set_next_window_size(imgui.ImVec2(viewport_size.x // 2, viewport_size.y))
        imgui.set_next_window_pos(imgui.ImVec2(viewport_size.x // 2, 0))
        imgui.set_next_window_bg_alpha(1.0)
        imgui.begin(
            "ImguiPlotter",
            flags=imgui.WindowFlags_.no_bring_to_front_on_focus | imgui.WindowFlags_.no_title_bar | imgui.WindowFlags_.no_decoration | imgui.WindowFlags_.no_resize | imgui.WindowFlags_.no_move,
        )
        # render the plotter's contents here
        pl.render_imgui()
        imgui.end()

        # GUI portion
        imgui.set_next_window_size(imgui.ImVec2(viewport_size.x // 2, viewport_size.y))
        imgui.set_next_window_pos(imgui.ImVec2(0, 0))
        imgui.set_next_window_bg_alpha(1.0)
        imgui.begin(
            "Controls",
            flags=imgui.WindowFlags_.no_bring_to_front_on_focus | imgui.WindowFlags_.no_resize | imgui.WindowFlags_.no_move,
        )

        imgui.text("Evaluation Environment Control")
        reset_clicked = imgui.button("Reset")
        if app_state.first_time or reset_clicked:
            app_state.eval_obs = app_state.eval_vecenv.reset()
        imgui.same_line()
        changed, app_state.play_mode = imgui.checkbox("Play Mode", app_state.play_mode)
        imgui.same_line()
        changed, app_state.deterministic = imgui.checkbox("Deterministic", app_state.deterministic)

        imgui.separator()

        imgui.text("Training Environment Control")
        imgui.text(f"# Parallel Environments: {app_state.train_vecenv.num_envs}")
        imgui.text(f"Rollout Length: {app_state.rollout_length}")
        imgui.text(f"Eval Reward Stat: {app_state.eval_rewards.sum(-1).mean():.2f} +/- {app_state.eval_rewards.sum(-1).std():.2f}")

        rollout_clicked = imgui.button("Rollout")
        imgui.same_line()
        learn_clicked = imgui.button("Learn")
        imgui.same_line()
        test_clicked = imgui.button("Test")

        # Iterating
        if app_state.iterating:
            rollout_clicked = True
            learn_clicked = True
            test_clicked = True

        if rollout_clicked:
            app_state.ppo.env.trajectory = []
            app_state.ppo.collect_rollouts(
                app_state.ppo.env,
                app_state.ppo_callback,
                app_state.ppo.rollout_buffer,
                n_rollout_steps=app_state.rollout_length,
            )
            app_state.trajectory_x = tree_stack(app_state.ppo.env.trajectory, 1)

        if learn_clicked:
            app_state.ppo.train()

        if test_clicked:
            app_state.eval_obs = app_state.eval_vecenv.reset()
            rewards = []
            for _ in range(app_state.rollout_length):
                action = app_state.ppo.policy.predict(app_state.eval_obs, deterministic=app_state.deterministic)[0]
                app_state.eval_obs, reward, done, _ = app_state.eval_vecenv.step(action)
                rewards.append(reward)
            app_state.eval_rewards = np.stack(rewards, -1)

        changed, app_state.n_iters = imgui.slider_int("# Iterations", app_state.n_iters, 1, 100)

        if app_state.iterating:
            app_state.iter_i += 1
            if app_state.iter_i >= app_state.n_iters:
                app_state.iterating = False
        iterate_clicked = imgui.button("Iterate")
        if iterate_clicked and not app_state.iterating:
            app_state.iterating = True
            app_state.iter_i = 0
        imgui.same_line()
        imgui.progress_bar(app_state.iter_i / app_state.n_iters, imgui.ImVec2(0, 0), f"{app_state.iter_i}/{app_state.n_iters}")

        changed, app_state.show_trails = imgui.checkbox("Show Trails", app_state.show_trails)

        imgui.text("Trail Color Code")
        cc_radio_clicked1 = imgui.radio_button("Body", app_state.color_code == 0)
        if cc_radio_clicked1:
            app_state.color_code = 0
        imgui.same_line()
        cc_radio_clicked2 = imgui.radio_button("Step Reward", app_state.color_code == 1)
        if cc_radio_clicked2:
            app_state.color_code = 1
        imgui.same_line()
        cc_radio_clicked3 = imgui.radio_button("Cumulative Reward", app_state.color_code == 2)
        if cc_radio_clicked3:
            app_state.color_code = 2
        imgui.same_line()
        cc_radio_clicked4 = imgui.radio_button("Estimated Value", app_state.color_code == 3)
        if cc_radio_clicked4:
            app_state.color_code = 3
        imgui.same_line()
        cc_radio_clicked5 = imgui.radio_button("Advantage", app_state.color_code == 4)
        if cc_radio_clicked5:
            app_state.color_code = 4

        cc_radio_clicked = np.any([cc_radio_clicked1, cc_radio_clicked2, cc_radio_clicked3, cc_radio_clicked4, cc_radio_clicked5])

        imgui.separator()

        imgui.text("Trajectory Browser")

        changed, app_state.traj_idx = imgui.slider_int("Trajectory Index", app_state.traj_idx, 0, app_state.train_vecenv.num_envs)
        changed, app_state.traj_frame = imgui.slider_int("Frame", app_state.traj_frame, 0, app_state.rollout_length - 1)

        imgui.end()

        rb_size = app_state.ppo.rollout_buffer.size()
        eval_state = app_state.eval_vecenv.state
        if app_state.play_mode:
            action = app_state.ppo.policy.predict(app_state.eval_obs, deterministic=app_state.deterministic)[0]
            app_state.eval_obs, reward, done, _ = app_state.eval_vecenv.step(action)

        else:
            # Guide and trails
            for i in range(app_state.train_vecenv.num_envs):
                for j in range(2):
                    if app_state.show_trails and rb_size > 0:
                        if rollout_clicked:
                            pos = np.array(app_state.trajectory_x.pos[i, :, j])
                            quat = np.array(app_state.trajectory_x.rot[i, :, j])
                            quat[..., [0, 1, 2, 3]] = quat[..., [1, 2, 3, 0]]
                            if j == 1:
                                offset = np.array([[0, 0, 0.6]]).repeat(quat.shape[0], 0)
                                pos += Rotation.from_quat(quat).apply(offset)
                            app_state.trail_meshes[i, j].points = pos
                            app_state.trail_meshes[i, j].lines = pv.MultipleLines(points=pos).lines

                        app_state.trail_actors[i, j].SetVisibility(True)
                    else:
                        app_state.trail_actors[i, j].SetVisibility(False)

        # Animating
        for i in range(len(app_state.scene_actors)):
            if app_state.play_mode or app_state.trajectory_x is None:
                pos = np.array(eval_state.pipeline_state.x.pos[0, i])
                quat = np.array(eval_state.pipeline_state.x.rot[0, i])
                quat[..., [0, 1, 2, 3]] = quat[..., [1, 2, 3, 0]]

                m = np.eye(4)
                m[:3, 3] = pos
                m[:3, :3] = Rotation.from_quat(quat).as_matrix()
                app_state.scene_actors[i].user_matrix = m
            else:
                pos = np.array(app_state.trajectory_x.pos[app_state.traj_idx, app_state.traj_frame, i])
                quat = np.array(app_state.trajectory_x.rot[app_state.traj_idx, app_state.traj_frame, i])
                quat[..., [0, 1, 2, 3]] = quat[..., [1, 2, 3, 0]]

                m = np.eye(4)
                m[:3, 3] = pos
                m[:3, :3] = Rotation.from_quat(quat).as_matrix()
                app_state.scene_actors[i].user_matrix = m

        # Coloring
        if app_state.show_trails and (app_state.iterating or cc_radio_clicked or rollout_clicked):
            for i in range(len(app_state.trail_meshes)):
                for j in range(2):
                    color = {
                        0: [1.0, 1.0, 1.0],
                        1: [1.0, 0.0, 0.0],
                    }[j]
                    colors = np.array([color]).repeat(app_state.trail_meshes[i, j].n_points, 0)

                    if app_state.color_code == 1:
                        colors = matplotlib.colormaps["viridis"](app_state.ppo.rollout_buffer.rewards[..., i])[..., :3]
                    elif app_state.color_code == 2:
                        colors = matplotlib.colormaps["viridis"](app_state.ppo.rollout_buffer.returns.reshape(app_state.rollout_length, -1)[..., i])[..., :3]
                    elif app_state.color_code == 3:
                        colors = matplotlib.colormaps["viridis"](app_state.ppo.rollout_buffer.values.reshape(app_state.rollout_length, -1)[..., i])[..., :3]
                    elif app_state.color_code == 4:
                        colors = matplotlib.colormaps["viridis"](app_state.ppo.rollout_buffer.advantages.reshape(app_state.rollout_length, -1)[..., i])[..., :3]

                    app_state.trail_meshes[i, j].point_data["color"] = colors

        app_state.first_time = False

    runner_params.callbacks.show_gui = gui
    runner_params.imgui_window_params.default_imgui_window_type = hello_imgui.DefaultImGuiWindowType.no_default_window
    immapp.run(runner_params=runner_params)


def main(args):
    env_name = "inverted_pendulum"
    mjcf_path = "brax/envs/assets/inverted_pendulum.xml"
    # env_name = "halfcheetah"
    # mjcf_path = "brax/envs/assets/half_cheetah.xml"

    backend = "mjx"
    batch_size = 1024
    episode_length = 256
    train_env_container = EnvContainer(env_name, backend, batch_size, True, episode_length)
    eval_env_container = EnvContainer(env_name, backend, 16, False, episode_length)
    train_vecenv = MyVecEnv(train_env_container, seed=0)
    eval_vecenv = MyVecEnv(eval_env_container, seed=0)

    if args.policy_path is not None:
        ppo = MyPPO.load(args.policy_path, train_vecenv)
    else:
        ppo = MyPPO(
            "MlpPolicy",
            train_vecenv,
            policy_kwargs={"log_std_init": -2, "net_arch": [64, 64]},
            learning_rate=3e-4,
            max_grad_norm=0.1,
            batch_size=16384,
            n_epochs=10,
            n_steps=episode_length,
        )
    pl = ImguiPlotter()
    plane_height = 0.0
    if env_name == "inverted_pendulum":
        plane_height = -0.5
    plane = pv.Plane(center=(0, 0, plane_height), direction=(0, 0, 1), i_size=10, j_size=10)

    pl.add_mesh(plane, show_edges=True)
    pl.add_axes()
    pl.camera.position = (0, -10, 0.1)
    pl.camera.focus = (0, 0, 0)
    pl.camera.up = (0, 0, 1)
    pl.enable_shadows()
    visual = XMLVisualDataContainer(mjcf_path)
    n = len(visual.meshes)
    scene_meshes = np.empty((n,), dtype=object)
    scene_actors = np.empty((n,), dtype=object)
    for j, mesh in enumerate(visual.meshes):
        # For inverted pendulum specifically: white for the cart, red for the pole
        if env_name == "inverted_pendulum":
            color = {
                0: [1.0, 1.0, 1.0],
                1: [1.0, 0.0, 0.0],
            }[j]
        else:
            color = [1.0, 1.0, 1.0]
        mesh.cell_data["color"] = np.array([color]).repeat(mesh.n_cells, 0)
        actor = pl.add_mesh(mesh, scalars="color", rgb=True, show_scalar_bar=False)
        scene_meshes[j] = mesh
        scene_actors[j] = actor
    scene_visuals = VisualArray(scene_meshes, scene_actors)

    trail_meshes = np.empty((batch_size, 2), dtype=object)
    trail_actors = np.empty((batch_size, 2), dtype=object)
    for i in range(batch_size):
        for j in range(2):
            color = {
                0: [1.0, 1.0, 1.0],
                1: [1.0, 0.0, 0.0],
            }[j]
            trail_mesh = pv.MultipleLines(points=np.zeros((2, 3)))
            trail_mesh.point_data["color"] = np.array([color]).repeat(trail_mesh.n_points, 0) * 1
            trail_actor = pl.add_mesh(trail_mesh, rgb=True, scalars="color", show_scalar_bar=False)
            trail_meshes[i, j] = trail_mesh
            trail_actors[i, j] = trail_actor
            trail_actor.SetVisibility(False)
        guide_mesh = pv.MultipleLines(points=np.zeros((3, 3)))
        guide_mesh.cell_data["color"] = np.array([[0.0, 0.0, 1.0]]).repeat(guide_mesh.n_cells, 0) * 1
        guide_actor = pl.add_mesh(guide_mesh, rgb=True, scalars="color", show_scalar_bar=False)
        guide_actor.SetVisibility(False)

    trail_visuals = VisualArray(trail_meshes, trail_actors)

    # Run the GUI
    app_state = AppState(scene_visuals, trail_visuals, train_vecenv, eval_vecenv, ppo)
    setup_and_run_gui(pl, app_state)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--policy_path", type=str)
    args = parser.parse_args()

    main(args)
