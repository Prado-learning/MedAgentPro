import argparse
import importlib
import json
import os
import sys
from typing import Any

from CodingAgent import Coding_Agent
from Decider import GPT_Decider, Pro_Decider
from Summary_Module import Summary_Module
from utils import build_qual_prompt, command_to_fn_name, ensure_pkg_inited, inputs_desc, read_prev_output, snake


DEFAULT_OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://chat.cloudapi.vip/v1")
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "chatgpt-4o-latest")


def resolve_data_root(data_root: str) -> tuple[str, str]:
    data_root_dir = os.path.abspath(data_root)
    package_name = os.path.basename(data_root_dir.rstrip(os.sep))
    package_parent = os.path.dirname(data_root_dir)
    if package_parent not in sys.path:
        sys.path.insert(0, package_parent)
    return data_root_dir, package_name


def load_json(json_path: str) -> Any:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def generated_fn_name(step: dict) -> str:
    base = step.get("action") or step.get("output_type") or "generated_fn"
    return f"{snake(base)}_{int(step.get('id', 0))}"


def build_requirement_and_name(step: dict, plan_by_id: dict, tool_by_id: dict, task_input_desc: str) -> tuple[str, str]:
    fn_name = generated_fn_name(step)
    in_desc_list = inputs_desc(step, plan_by_id, tool_by_id, task_input_desc)
    in_desc_str = ", ".join(in_desc_list)
    out_desc = str(step.get("output_type", "")).strip()
    step_id = int(step.get("id", 0))

    requirement = (
        "Implement a Python function with the EXACT signature:\n"
        f"{fn_name}(inputs, save_dir, save_name)\n\n"
        "Semantics:\n"
        "- `inputs` is a LIST; its elements correspond IN ORDER to the step dependencies.\n"
        f"- Conceptual inputs: {in_desc_str}\n"
        f"- Output (conceptual): {out_desc}\n\n"
        "Constraints:\n"
        "- The function is self-contained; add imports inside if needed. No print statements.\n"
        "- Always use `os.path.join(save_dir, save_name)` as the ONLY output file path for any NON-IMAGE result.\n"
        "- NON-IMAGE includes text/json/metrics/numerical values/tables/etc. Do NOT create separate text/json files.\n"
        "- If the output is an IMAGE (extensions: .png/.jpg/.jpeg/.tif/.tiff/.bmp/.gif/.webp), you may write that image to disk using save_dir/save_name as the filename.\n"
        "- When writing NON-IMAGE results, open/create the JSON file at `os.path.join(save_dir, save_name)`, read existing JSON if present, "
        f"update a unique key for this step (for example 'step_{step_id}'), and write back with UTF-8 and ensure_ascii=False.\n"
        "- Add a brief docstring explaining inputs (list), outputs, and side effects.\n"
    )
    return fn_name, requirement


def build_tool_registry(package_name: str, toolset: list[dict[str, Any]]) -> dict[str, Any]:
    tool_module = importlib.import_module(f"{package_name}.tools")
    tool_module = importlib.reload(tool_module)
    registry: dict[str, Any] = {}
    for tool in toolset:
        fn_name = command_to_fn_name(tool.get("command", ""))
        if fn_name and hasattr(tool_module, fn_name):
            registry[fn_name] = getattr(tool_module, fn_name)
    return registry


def register_generated_function(package_name: str, registry: dict[str, Any], fn_name: str) -> bool:
    module_name = f"{package_name}.tools.GenCode"
    importlib.invalidate_caches()
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        return False
    module = importlib.reload(module)
    if not hasattr(module, fn_name):
        return False
    registry[fn_name] = getattr(module, fn_name)
    return True


def ensure_codegen_file(data_root_dir: str) -> str:
    code_path = os.path.join(data_root_dir, "tools", "GenCode.py")
    os.makedirs(os.path.dirname(code_path), exist_ok=True)
    if not os.path.exists(code_path):
        with open(code_path, "w", encoding="utf-8") as f:
            f.write("# Generated code\n")
    return code_path


def is_coding_step(step: dict, tool_by_id: dict) -> bool:
    tool_ids = step.get("tool", []) or []
    if not isinstance(tool_ids, list):
        tool_ids = [tool_ids]
    return any("coding" in str(tool_by_id.get(int(tid), {}).get("type", "")).lower() for tid in tool_ids)


def ensure_generated_functions(
    data_root_dir: str,
    package_name: str,
    task: dict,
    plan: list[dict[str, Any]],
    tool_by_id: dict[int, dict[str, Any]],
    plan_by_id: dict[int, dict[str, Any]],
    registry: dict[str, Any],
    api_key: str | None,
    base_url: str | None,
    model: str | None,
) -> str:
    ensure_pkg_inited(data_root_dir)
    code_path = ensure_codegen_file(data_root_dir)
    coder = None

    for step in plan:
        if not is_coding_step(step, tool_by_id):
            continue

        fn_name, requirement = build_requirement_and_name(
            step=step,
            plan_by_id=plan_by_id,
            tool_by_id=tool_by_id,
            task_input_desc=task.get("input", ""),
        )

        if register_generated_function(package_name, registry, fn_name):
            continue

        if not api_key:
            raise RuntimeError(
                f"Missing generated function '{fn_name}' and OPENAI_API_KEY is empty. "
                "Set OPENAI_API_KEY or pre-create the generated coding function."
            )

        if coder is None:
            coder = Coding_Agent(api_key, base_url=base_url, model=model)

        coder.generate_function(
            output_file=code_path,
            requirement=requirement,
            enforce_function_name=fn_name,
            extra_context=(
                "`inputs` is a list; each item may be a file path or an in-memory object. "
                "If an input is a binary mask image loaded from disk, handle both 0/1 and 0/255 pixel conventions."
            ),
            model=model,
        )

        if not register_generated_function(package_name, registry, fn_name):
            raise RuntimeError(f"Generated function '{fn_name}' was not found after writing {code_path}.")

    return code_path


def resolve_step_inputs(step: dict, plan_by_id: dict[int, dict[str, Any]], image_path: str, save_dir: str) -> list[str]:
    resolved = []
    for dep in step.get("input_type", []) or []:
        dep = int(dep)
        if dep == 0:
            resolved.append(image_path)
            continue
        prev = plan_by_id.get(dep)
        if prev is None:
            raise KeyError(f"Step {step.get('id')} depends on missing step id {dep}.")
        prev_save = prev.get("output_path", "")
        resolved.append(os.path.join(save_dir, prev_save))
    return resolved


def run_quantitative_step(
    step: dict,
    image_path: str,
    save_dir: str,
    tool_by_id: dict[int, dict[str, Any]],
    plan_by_id: dict[int, dict[str, Any]],
    registry: dict[str, Any],
) -> None:
    tool_ids = step.get("tool", []) or []
    if not isinstance(tool_ids, list):
        tool_ids = [tool_ids]
    save_name = step.get("output_path", "")
    resolved_inputs = resolve_step_inputs(step, plan_by_id, image_path, save_dir)

    for tid in tool_ids:
        tool = tool_by_id.get(int(tid))
        if tool is None:
            raise KeyError(f"Tool id {tid} not found for step {step.get('id')}.")

        if "coding" in str(tool.get("type", "")).lower():
            fn_name = generated_fn_name(step)
        else:
            fn_name = command_to_fn_name(tool.get("command", ""))

        fn = registry.get(fn_name)
        if fn is None:
            raise RuntimeError(f"Function '{fn_name}' is not registered for step {step.get('id')}.")

        if "coding" in str(tool.get("type", "")).lower():
            fn(resolved_inputs, save_dir, save_name)
        else:
            primary_input = resolved_inputs[0] if resolved_inputs else image_path
            fn(primary_input, save_dir, save_name)


def run_qualitative_step(
    step: dict,
    image_path: str,
    save_dir: str,
    plan_by_id: dict[int, dict[str, Any]],
    analyzer: GPT_Decider,
    summarizer: Summary_Module,
) -> None:
    image_input: list[str] = []
    text_input: list[str] = []
    step_id = int(step.get("id", 0))
    save_name = step.get("output_path", "")

    for dep in step.get("input_type", []) or []:
        dep = int(dep)
        if dep == 0:
            image_input.append(image_path)
            continue
        prev_step = plan_by_id.get(dep, {})
        prev_save_name = prev_step.get("output_path", "")
        text_value, image_value = read_prev_output(save_dir, prev_save_name, dep)
        if image_value:
            image_input.append(image_value)
        if text_value:
            text_input.append(text_value)

    question = step.get("action", "")
    full_prompt = build_qual_prompt(question, text_input)
    diagnosis_path = os.path.join(save_dir, save_name)

    analyzer.decide(
        output_file=diagnosis_path,
        prompt=full_prompt,
        image_paths=image_input,
        field=f"step_{step_id}",
    )

    summary_prompt = (
        f"Based on the above text, please provide a brief summary. "
        f"The task is: {question}. Does this patient have the abnormal?"
    )
    summarizer.summarize(
        input_file=diagnosis_path,
        output_file=os.path.join(save_dir, "brief_diagnosis.json"),
        prompt=summary_prompt,
        field=f"step_{step_id}",
    )


def finalize_case(
    task: dict,
    plan: list[dict[str, Any]],
    save_dir: str,
    decider: Pro_Decider,
) -> dict[str, Any]:
    input_desc = str(task.get("input", "")).strip()
    disease_goal = str(task.get("disease", "")).strip()
    brief_path = os.path.join(save_dir, "brief_diagnosis.json")

    if os.path.exists(brief_path):
        brief_data = load_json(brief_path)
    else:
        brief_data = {}

    indicators = []
    for step in plan:
        if str(step.get("output_type", "")).lower() != "final indicator":
            continue
        indicators.append(
            {
                "indicator_name": step.get("action", ""),
                "if_abnormal": brief_data.get(f"step_{step.get('id')}", {}),
            }
        )

    if not indicators:
        raise RuntimeError("No final indicators were produced for the single-case demo.")

    decide_prompt = (
        "You are a clinical decision assistant.\n"
        "Task and context:\n"
        f"- Input: {input_desc}\n"
        f"- Goal: {disease_goal}\n\n"
        "Please propose reasonable weights (sum to 1) and a threshold in [0,1]. "
        "Return ONLY a JSON object with the keys: 'weights' (list of {'indicator_name','weight'}), "
        "'threshold' (float), and an optional 'notes' (short string)."
    )

    return decider.decide(
        output_file=os.path.join(save_dir, "final_diagnosis.json"),
        prompt=decide_prompt,
        indicators=indicators,
        field="overall",
    )


def default_save_dir(data_root_dir: str, image_path: str) -> str:
    stem = os.path.splitext(os.path.basename(image_path))[0]
    parent = os.path.basename(os.path.dirname(image_path))
    case_name = f"{parent}_{stem}" if parent else stem
    return os.path.join(data_root_dir, "record", case_name)


def run_single_case_demo(
    image_path: str,
    data_root: str = "Glaucoma",
    save_dir: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    image_path = os.path.abspath(image_path)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    data_root_dir, package_name = resolve_data_root(data_root)
    if not os.path.isdir(data_root_dir):
        raise FileNotFoundError(f"Data root not found: {data_root_dir}")

    task = load_json(os.path.join(data_root_dir, "task.json"))
    plan = load_json(os.path.join(data_root_dir, "plan.json"))
    toolset = load_json(os.path.join(data_root_dir, "toolset.json"))

    tool_by_id = {int(t["id"]): t for t in toolset if "id" in t}
    plan_by_id = {int(s["id"]): s for s in plan if "id" in s}

    registry = build_tool_registry(package_name, toolset)

    api_key = api_key or os.getenv("OPENAI_API_KEY")
    base_url = base_url or DEFAULT_OPENAI_BASE_URL
    model = model or DEFAULT_OPENAI_MODEL

    code_path = ensure_generated_functions(
        data_root_dir=data_root_dir,
        package_name=package_name,
        task=task,
        plan=plan,
        tool_by_id=tool_by_id,
        plan_by_id=plan_by_id,
        registry=registry,
        api_key=api_key,
        base_url=base_url,
        model=model,
    )

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is empty. Set it in the environment or pass --api-key.")

    analyzer = GPT_Decider(api_key, base_url=base_url, model=model)
    summarizer = Summary_Module(api_key, base_url=base_url, model=model)
    decider = Pro_Decider(api_key, base_url=base_url, model=model)

    save_dir = os.path.abspath(save_dir) if save_dir else default_save_dir(data_root_dir, image_path)
    os.makedirs(save_dir, exist_ok=True)

    for step in plan:
        action_type = str(step.get("action_type", "")).lower()
        if action_type == "quantitative":
            run_quantitative_step(
                step=step,
                image_path=image_path,
                save_dir=save_dir,
                tool_by_id=tool_by_id,
                plan_by_id=plan_by_id,
                registry=registry,
            )
        elif action_type == "qualitative":
            run_qualitative_step(
                step=step,
                image_path=image_path,
                save_dir=save_dir,
                plan_by_id=plan_by_id,
                analyzer=analyzer,
                summarizer=summarizer,
            )
        else:
            raise ValueError(f"Unsupported action_type '{action_type}' in step {step.get('id')}.")

    final_result = finalize_case(
        task=task,
        plan=plan,
        save_dir=save_dir,
        decider=decider,
    )

    return {
        "image_path": image_path,
        "save_dir": save_dir,
        "generated_code_path": code_path,
        "diagnosis_path": os.path.join(save_dir, "diagnosis.json"),
        "brief_diagnosis_path": os.path.join(save_dir, "brief_diagnosis.json"),
        "final_diagnosis_path": os.path.join(save_dir, "final_diagnosis.json"),
        "overall": final_result.get("overall"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MedAgent-Pro single-sample glaucoma demo.")
    parser.add_argument("--image", required=True, help="Path to a single input image.")
    parser.add_argument("--data-root", default="Glaucoma", help="Task package root, default: Glaucoma")
    parser.add_argument("--output-dir", default=None, help="Optional output directory for this sample.")
    parser.add_argument("--api-key", default=None, help="Optional OpenAI API key. Defaults to OPENAI_API_KEY.")
    parser.add_argument("--base-url", default=None, help="Optional OpenAI-compatible base URL.")
    parser.add_argument("--model", default=None, help="Optional model name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_single_case_demo(
        image_path=args.image,
        data_root=args.data_root,
        save_dir=args.output_dir,
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
