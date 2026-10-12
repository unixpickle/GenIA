# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Compatibility imports; implementations live in ``genia.core.utils.final_rendering``."""

from genia.core.utils.final_rendering import runtime as _runtime
from genia.core.utils.final_rendering.assets import (
    Camera,
    ObjectAssets,
    Tracks,
    Transform,
    _obj_xyz_members,
)
from genia.core.utils.final_rendering.builtins import (
    register_builtin_renderers as _register_builtin_renderers,
)
from genia.core.utils.final_rendering.cameras import (
    TestView,
    _CameraSequence,
    _rebase_c2w,
    _src_frame,
    resolve_test_cameras,
)
from genia.core.utils.final_rendering.cli import (
    build_parser,
    main,
    parse_bg,
    parse_index_spec,
    parse_resolution,
    process_run,
    render_menu,
    viz_roots,
)
from genia.core.utils.final_rendering.configuration import (
    _WORLD_UP,
    _WORLD_UP_DATASETS,
    has_dataset_world_up,
    load_run_config,
    resolve_up,
)
from genia.core.utils.final_rendering.contact_sheet import (
    render_contact_sheet,
)
from genia.core.utils.final_rendering.context import (
    RenderContext,
)
from genia.core.utils.final_rendering.deformation import (
    DeformationSource,
    _warp_kwargs,
    _WarpShim,
)
from genia.core.utils.final_rendering.discovery import (
    _TIMESTAMP_RE,
    _UNKNOWN,
    RunPath,
    discover_final_dirs,
    parse_run_path,
)
from genia.core.utils.final_rendering.geometry import (
    _MESH_CACHE,
    GaussianCloud,
    MeshData,
    _cam_to_world_points,
    _sim3_matrix,
    _world_to_cam_points,
    apply_color_shift,
    apply_sim3,
    apply_sim3_points,
    blend_geometry,
    cam_to_world,
    concat_clouds,
    load_gaussian_ply,
    load_mesh,
    merge_meshes,
    p3d_to_r3,
    r3_to_p3d_points,
    rows_correspond,
    unapply_sim3_points,
)
from genia.core.utils.final_rendering.images import (
    _supersampled_size,
    _write_cropped,
    alpha_bbox,
    fit_to_native,
    flatten_alpha,
    sequence_crop_box,
    to_rgba,
)
from genia.core.utils.final_rendering.options import (
    _FIGURE_CROP_OPTIONS,
    _TRACK_DRAW_OPTIONS,
    _TRACK_OPTIONS,
    _TRAIN_CAMERA_OPTIONS,
)
from genia.core.utils.final_rendering.orbit import (
    motion_sample,
    render_orbit,
    sample_of,
)
from genia.core.utils.final_rendering.projection import (
    _GLCTX,
    bounding_points,
    bounds_aabb,
    centred_intrinsics,
    fit_focal,
    fit_square_focal,
    project_r3,
    render_cloud,
    render_mesh,
    scale_intrinsics,
    square_intrinsics,
    trim_outliers,
)
from genia.core.utils.final_rendering.registry import (
    RENDERERS,
    Option,
    Renderer,
    _coerce,
    _split_opt,
    check_options_are_claimed,
    renderer,
    resolve_options,
)
from genia.core.utils.final_rendering.run import (
    FinalRun,
)
from genia.core.utils.final_rendering.runtime import (
    _lazy_imports,
    log,
)
from genia.core.utils.final_rendering.tracks import (
    _ANCHOR_K,
    _ANCHOR_SPARSITY,
    _TRACK_HEAD_PX,
    _anchor_keep_mask,
    _track_trails,
    draw_tracks,
    farthest_point_indices,
    overlay_trails,
    track_colors,
)
from genia.core.utils.final_rendering.views import (
    _render_from_train_cameras,
    render_synth_nvs,
    render_track_overlay,
    render_track_overlay_nvs,
    render_track_overlay_test,
    render_train_views,
)
from genia.core.utils.final_rendering.world import (
    _AXIS_COLORS,
    draw_pose_axes,
    load_background_cloud,
    overview_cameras,
    render_world_space,
    view_tag,
)


def __getattr__(name):
    if name in {
        "Image", "torch", "render_gaussian_params",
        "bulk_keep_mask", "visible_bulk_mask", "oversize_splat_mask", "connected_bulk_mask",
        "p3d_to_r3_positions", "p3d_to_r3_quaternions",
        "quaternion_invert", "quaternion_multiply", "quaternion_to_matrix",
    }:
        return getattr(_runtime, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    raise SystemExit(main())
