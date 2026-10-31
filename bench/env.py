"""Record the machine this project's numbers come from: `modal run modal_app.py --cmd bench.env`."""

from bench.common import collect_env, save_result

if __name__ == "__main__":
    env = collect_env()
    print(env["nvidia_smi"])
    save_result("env", {})
