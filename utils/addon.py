import sys
import pathlib
import addon_utils


# Searches for fast64 and appends it to sys.path to allow importing functions
def addon_set_fast64_path():
    for mod in addon_utils.modules():
        if mod.bl_info.get("name") == "Fast64":
            f64_path = pathlib.Path(mod.__file__).parent
            sys.path.append(str(f64_path))
            return

    raise RuntimeError("Fast64 not found")
