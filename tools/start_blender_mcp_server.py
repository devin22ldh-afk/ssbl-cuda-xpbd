import importlib.util
import pathlib

import bpy

addon_path = pathlib.Path(
    r"C:\Users\Administrator\AppData\Roaming\Blender Foundation\Blender\5.0\scripts\addons\blender-mcp\__init__.py"
)
spec = importlib.util.spec_from_file_location("blender_mcp_runtime_addon", addon_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

module.register()
server = module.BlenderMCPServer(host="127.0.0.1", port=9876)
server.start()
bpy.app.driver_namespace["ssbl_mcp_server"] = server

print("SSBL_MCP_SERVER_READY 127.0.0.1:9876")
