import asyncio
from dotenv import load_dotenv
load_dotenv()
from e2b_code_interpreter import Sandbox, AsyncSandbox

_GLOBAL_SANDBOX = None

async def async_get_or_create_sandbox():
    global _GLOBAL_SANDBOX
    if _GLOBAL_SANDBOX is None:
        _GLOBAL_SANDBOX = await AsyncSandbox.create("OpenEvalsPython")
    return _GLOBAL_SANDBOX
async def main():
    sbx = await async_get_or_create_sandbox()
    execution = await sbx.run_code("print('hello world')") # Execute Python inside the sandbox
    print(execution.logs)

    files = await sbx.files.list("/")
    print(files)

if __name__ == "__main__":
    asyncio.run(main())