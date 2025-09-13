import gradio as gr
import requests
import json
import random
from typing import Any, Dict, List, Optional


DEFAULT_OLLAMA_URL = "http://localhost:11434"


def check_ollama_connection(ollama_url: str) -> bool:
    try:
        r = requests.get(f"{ollama_url}/api/tags", timeout=5)
        r.raise_for_status()
        return True
    except requests.RequestException:
        return False


def get_ollama_models(ollama_url: str) -> List[str]:
    try:
        r = requests.get(f"{ollama_url}/api/tags", timeout=5)
        r.raise_for_status()
        data = r.json()
        return [m.get("name", "") for m in data.get("models", [])]
    except requests.RequestException:
        return []


def pick_default_model(models: List[str]) -> str:
    for cand in ["exaone3.5", "exaone:latest", "exaone"]:
        if cand in models:
            return cand
    for m in models:
        if "exaone" in m.lower():
            return m
    return models[0] if models else ""


def generate_once(
    ollama_url: str,
    model: str,
    prompt: str,
    options: Optional[Dict[str, Any]] = None,
    timeout_s: Any = (3, 15),
) -> Optional[str]:
    merged_opts: Dict[str, Any] = {"num_predict": 256}
    if options:
        try:
            merged_opts.update({k: v for k, v in options.items() if v is not None})
        except Exception:
            pass
    payload = {"model": model, "prompt": prompt, "stream": False, "options": merged_opts}
    try:
        r = requests.post(f"{ollama_url}/api/generate", json=payload, timeout=timeout_s)
        r.raise_for_status()
        data = r.json()
        return data.get("response", "")
    except requests.RequestException:
        return None

def stream_generate(
    ollama_url: str,
    model: str,
    prompt: str,
    options: Optional[Dict[str, Any]] = None,
    # Short connect/read timeouts to improve cancel responsiveness
    timeout_s: Any = (3, 5),
):
    # Merge conservative defaults to avoid very long generations
    merged_opts: Dict[str, Any] = {"num_predict": 256}
    if options:
        try:
            merged_opts.update({k: v for k, v in options.items() if v is not None})
        except Exception:
            pass
    payload = {"model": model, "prompt": prompt, "options": merged_opts}
    try:
        with requests.post(
            f"{ollama_url}/api/generate",
            json=payload,
            stream=True,
            # Accept tuple timeouts (connect, read) for better responsiveness
            timeout=timeout_s,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    j = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError:
                    continue
                if "response" in j:
                    yield j["response"]
                if j.get("done"):
                    break
    except requests.RequestException:
        return


def build_agent_prompt(
    role_title: str,
    stance_instruction: str,
    system_rules: str,
    user_question: str,
    memory_text: str,
    depth: int,
    extra_directive: str = "",
) -> str:
    return (
        "한국어로 응답하세요.\n"
        f"당신은 '{role_title}' 입니다. {stance_instruction}\n\n"
        f"규칙:\n{system_rules}\n\n"
        f"질문:\n{user_question}\n\n"
        f"대화 메모(요약):\n{memory_text}\n\n"
        f"요청:\n- 직전 상대 발언을 한 줄로 요약\n- 핵심 주장을 2문장 이내로 명확히\n- 필요할 때만 코드 블록 포함(없어도 됨)\n- 공손하지만 단호하게 상호작용적으로 응답\n- 깊이 레벨: {depth}\n{extra_directive}\n"
    )


def trim_memory(convo: List[Dict[str, str]], limit_chars: int = 2000) -> str:
    text = ""
    total = 0
    for msg in reversed(convo):
        line = f"{msg.get('name', msg.get('role',''))}: {msg.get('content','')}\n"
        total += len(line)
        if total > limit_chars:
            break
        text = line + text
    return text.strip()


def choose_fixed_stance(user_question: str) -> str:
    q = (user_question or "").lower()
    mapping = [
        ("greedy", "그리디"),
        ("그리디", "그리디"),
        ("dynamic", "다이나믹 프로그래밍"),
        ("다이나믹", "다이나믹 프로그래밍"),
        ("동적", "다이나믹 프로그래밍"),
        ("dp", "다이나믹 프로그래밍"),
        ("divide", "분할 정복"),
        ("분할", "분할 정복"),
        ("backtracking", "백트래킹"),
        ("백트", "백트래킹"),
        ("bfs", "탐색(BFS/DFS)"),
        ("dfs", "탐색(BFS/DFS)"),
        ("탐색", "탐색(BFS/DFS)"),
        ("hash", "해시 기반"),
        ("해시", "해시 기반"),
    ]
    for key, stance in mapping:
        if key in q:
            return stance
    candidates = [
        "그리디",
        "다이나믹 프로그래밍",
        "분할 정복",
        "백트래킹",
        "탐색(BFS/DFS)",
        "해시 기반",
    ]
    return random.choice(candidates)


def main():
    with gr.Blocks(css='''
        .gradio-container{background:linear-gradient(180deg,#fdf2f8 0%,#eef2ff 100%)}
        .btn-wide button{width:100%; border-radius:12px; padding:10px 12px; background:linear-gradient(90deg,#6366f1,#22d3ee); color:#fff; border:none}
        .btn-wide button:hover{filter:brightness(1.06)}
        .panel{padding:14px;border-radius:14px;border:1px solid #e5e7eb;box-shadow:0 6px 20px rgba(0,0,0,.06)}
        .panel h3{margin-top:0}
        .panel textarea, .panel input{border-radius:10px !important; border:1px solid #e5e7eb !important}
        .chatbox{background:#fff;border:1px solid #e5e7eb;border-radius:14px;box-shadow:0 4px 14px rgba(0,0,0,.06)}
    ''') as demo:
        models0 = get_ollama_models(DEFAULT_OLLAMA_URL)
        default_model = pick_default_model(models0)

        with gr.Row():
            ollama_url = gr.Textbox(label="Ollama URL", value=DEFAULT_OLLAMA_URL)
            model_dd = gr.Dropdown(label="Model", choices=models0, value=default_model)
            fast_mode = gr.Checkbox(label="Fast mode (no streaming)", value=True)
            start_btn = gr.Button("Start")
            stop_btn = gr.Button("Stop", interactive=False)
            reset_btn = gr.Button("Reset")
            status = gr.Textbox(label="Status", interactive=False)

        # Left stacked panels; right topic+chat
        with gr.Row():
            with gr.Column(scale=1, elem_classes=["panel"]):
                gr.Markdown("### 🟣 Greedy Rebel 설정")
                agent1_name = gr.Textbox(label="이름", value="Greedy Rebel")
                agent1_system = gr.TextArea(
                    label="시스템 프롬프트",
                    value=("주제에 맞춰 한 가지 입장을 스스로 정하고,\n"
                           "토론 내내 그 입장만 고수하라(전환 금지).\n"
                           "필요할 때만 간결한 코드로 설득하라."),
                    lines=5,
                )
                agent1_mem = gr.Slider(label="Memory Size (chars)", minimum=200, maximum=10000, value=2000, step=100)

                gr.Markdown("### 🔵 DP Master 설정")
                agent2_name = gr.Textbox(label="이름", value="DP Master")
                agent2_system = gr.TextArea(
                    label="시스템 프롬프트",
                    value=("주제에 대해 구조적이고 검증 가능한 관점을 제시하라.\n"
                           "특정 기법에 집착하지 말고, 상황에 맞게 반대 입장을 취하라."),
                    lines=5,
                )
                agent2_mem = gr.Slider(label="Memory Size (chars)", minimum=200, maximum=10000, value=2000, step=100)

            with gr.Column(scale=2, elem_classes=["panel"]):
                user_question = gr.Textbox(label="📝 토론 주제", value="그리디와 다이나믹 프로그래밍, 어느 쪽이 더 좋을까?")
                chat = gr.Chatbot(label="🗨️ 인터랙티브 토론", height=640, type="messages", elem_classes=["chatbox"])
                with gr.Row(equal_height=True):
                    btn_style = {"elem_classes": ["btn-wide"]}
                    opt_1 = gr.Button("1번 의견 심화", visible=False, **btn_style)
                    opt_2 = gr.Button("2번 의견 심화", visible=False, **btn_style)
                    opt_3 = gr.Button("계속 토론", visible=False, **btn_style)
                    opt_4 = gr.Button("다른 예제로 비교", visible=False, **btn_style)
                    opt_5 = gr.Button("새로운 질문", visible=False, **btn_style)

        # Hidden rules default
        system_rules = gr.State("- 서로 반대 입장을 유지\n- 핵심만 간결히, 필요할 때만 코드\n- 비방/모욕 금지, 건설적 토론\n- 사용자가 선택한 흐름을 따르기")

        state = gr.State({
            "running": False,
            "history": [],
            "convo": [],
            "depth": 1,
            "last_choice": None,
            "agent_index": None,
            "agent1_fixed_stance": None,
            "agents": [
                {"name": "Greedy Rebel", "role_title": "Greedy Rebel", "system": "", "temperature": 0.9, "top_k": 40, "memory_size": 2000},
                {"name": "DP Master", "role_title": "DP Master", "system": "", "temperature": 0.9, "top_k": 40, "memory_size": 2000},
            ],
        })

        def on_url_change(url: str):
            models = get_ollama_models(url)
            return gr.Dropdown.update(choices=models, value=pick_default_model(models))

        ollama_url.change(on_url_change, inputs=[ollama_url], outputs=[model_dd])

        def on_start(q: str, rules: str, url: str, model: str, fast: bool,
                     a1_name: str, a1_sys: str, a1_mem: int,
                     a2_name: str, a2_sys: str, a2_mem: int,
                     st: Dict[str, Any]):
            if st.get("running"):
                opts = [gr.update(visible=True)] * 5
                return (st["history"], gr.update(interactive=False), gr.update(interactive=True), *opts, "이미 진행 중")

            if not check_ollama_connection(url):
                opts = [gr.update(visible=False)] * 5
                return (st["history"], gr.update(interactive=True), gr.update(interactive=False), *opts, "Ollama 연결 실패")

            st["running"] = True
            st["history"] = []
            st["convo"] = []
            st["depth"] = 1
            st["last_choice"] = None
            st["agent_index"] = random.randint(0, 1)

            st["agents"][0].update({
                "name": (a1_name or "Greedy Rebel").strip(),
                "system": a1_sys.strip(),
                "memory_size": int(a1_mem),
            })
            st["agents"][1].update({
                "name": (a2_name or "DP Master").strip(),
                "system": a2_sys.strip(),
                "memory_size": int(a2_mem),
            })

            st["agent1_fixed_stance"] = choose_fixed_stance(q)

            mem = trim_memory(st["convo"], 1600)
            a1 = st["agents"][0]
            fixed = st.get("agent1_fixed_stance", "그리디")
            p1 = build_agent_prompt(
                a1["role_title"], f"{a1['system']}\n이번 토론의 고정 입장: '{fixed}'. 이 입장만 유지하여 답하라.",
                rules, q, mem, st["depth"],
                extra_directive=f"'{fixed}' 관점과 일관된 경우에만 코드 포함(필요 시).",
            )

            a2 = st["agents"][1]
            p2 = build_agent_prompt(
                a2["role_title"], f"{a2['system']}\n직전 발언에 대해 상호작용적으로 반대 입장을 제시하라.",
                rules, q, mem, st["depth"],
            )

            start_dis = gr.update(interactive=False)
            stop_en = gr.update(interactive=True)
            opts_hidden = [gr.update(visible=False)] * 5
            status_text = f"토론 시작 · 고정 입장: {fixed}"

            # Agent 1
            full1 = ""
            st["history"].append({"role": "user", "content": f"{a1['name']}: "})
            st["convo"].append({"name": a1["name"], "content": ""})
            yield (st["history"], start_dis, stop_en, *opts_hidden, status_text)
            try:
                if fast:
                    _res1 = generate_once(url, model, p1, options={"temperature": a1.get("temperature", 0.9), "top_k": a1.get("top_k", 40)})
                    chunk_iter = [(_res1 or "")]
                else:
                    chunk_iter = stream_generate(url, model, p1, options={"temperature": a1.get("temperature", 0.9), "top_k": a1.get("top_k", 40)})
                for chunk in chunk_iter:
                    if not st.get("running"):
                        break
                    full1 += chunk
                    st["history"][-1]["content"] = f"{a1['name']}: {full1}"
                    st["convo"][-1]["content"] = full1
                    yield (st["history"], start_dis, stop_en, *opts_hidden, f"{a1['name']} 타이핑 중...")
            except Exception:
                st["history"][-1]["content"] = f"{a1['name']}: (생성 중 오류)"
                st["convo"][-1]["content"] = "(생성 중 오류)"

            # Agent 2
            full2 = ""
            st["history"].append({"role": "assistant", "content": f"{a2['name']}: "})
            st["convo"].append({"name": a2["name"], "content": ""})
            yield (st["history"], start_dis, stop_en, *opts_hidden, f"{a2['name']} 타이핑 중...")
            try:
                if fast:
                    _res2 = generate_once(url, model, p2, options={"temperature": a2.get("temperature", 0.9), "top_k": a2.get("top_k", 40)})
                    chunk_iter2 = [(_res2 or "")]
                else:
                    chunk_iter2 = stream_generate(url, model, p2, options={"temperature": a2.get("temperature", 0.9), "top_k": a2.get("top_k", 40)})
                for chunk in chunk_iter2:
                    if not st.get("running"):
                        break
                    full2 += chunk
                    st["history"][-1]["content"] = f"{a2['name']}: {full2}"
                    st["convo"][-1]["content"] = full2
                    yield (st["history"], start_dis, stop_en, *opts_hidden, f"{a2['name']} 타이핑 중...")
            except Exception:
                st["history"][-1]["content"] = f"{a2['name']}: (생성 중 오류)"
                st["convo"][-1]["content"] = "(생성 중 오류)"

            yield (st["history"], start_dis, stop_en, *[gr.update(visible=True)]*5, "다음 진행을 선택하세요")

        def on_stop(st: Dict[str, Any]):
            st["running"] = False
            return (st["history"], gr.update(interactive=True), gr.update(interactive=False), *[gr.update(visible=True)]*5, "정지됨")

        def on_reset(st: Dict[str, Any]):
            st.update({"running": False, "history": [], "convo": [], "depth": 1, "last_choice": None, "agent_index": None})
            return ([], gr.update(interactive=True), gr.update(interactive=False), *[gr.update(visible=False)]*5, "초기화 완료")

        def proceed(choice: str, q: str, rules: str, url: str, model: str, st: Dict[str, Any]):
            if not st.get("running"):
                st["running"] = True
            st["last_choice"] = choice
            st["depth"] = max(1, st.get("depth", 1)) + 1

            a1 = st["agents"][0]
            a2 = st["agents"][1]
            mem = trim_memory(st["convo"], 2000)

            # Quick connectivity check to avoid long hangs
            if not check_ollama_connection(url):
                return (
                    st["history"],
                    gr.update(interactive=True),
                    gr.update(interactive=False),
                    *[gr.update(visible=True)] * 5,
                    "Ollama 연결 실패 (다시 시도하세요)",
                )

            if choice == "opt1":
                fixed = st.get("agent1_fixed_stance", "고정 입장")
                directive = f"직전 발언에 대해 '{fixed}' 관점으로 반박하고, 필요할 때만 코드 포함"
                prompt = build_agent_prompt(a1["role_title"], f"{a1['system']}\n항상 '{fixed}' 입장만 유지.", rules, q, mem, st["depth"], directive)
                speaker = a1
            elif choice == "opt2":
                directive = "직전 발언에 대해 반대 입장을 상호작용적으로 제시하고, 필요할 때만 코드 포함"
                prompt = build_agent_prompt(a2["role_title"], a2["system"], rules, q, mem, st["depth"], directive)
                speaker = a2
            elif choice == "opt3":
                fixed = st.get("agent1_fixed_stance", "���� ����")
                p1 = build_agent_prompt(
                    a1["role_title"], f"{a1['system']}\n�׻� '{fixed}' ���常 ����.",
                    rules, q, mem, st["depth"],
                    extra_directive=f"'{fixed}' ������ �ϰ��� ��쿡�� �ڵ� ����(�ʿ� ��).",
                )
                p2 = build_agent_prompt(
                    a2["role_title"], a2["system"],
                    rules, q, mem, st["depth"],
                )

                start_dis = gr.update(interactive=False)
                stop_en = gr.update(interactive=True)
                opts_hidden = [gr.update(visible=False)] * 5

                # Agent 1
                full1 = ""
                st["history"].append({"role": "user", "content": f"{a1['name']}: "})
                st["convo"].append({"name": a1["name"], "content": ""})
                yield (st["history"], start_dis, stop_en, *opts_hidden, f"{a1['name']} Ÿ���� ��...")
                try:
                    for chunk in stream_generate(url, model, p1, options={"temperature": a1.get("temperature", 0.9), "top_k": a1.get("top_k", 40)}):
                        if not st.get("running"):
                            break
                        full1 += chunk
                        st["history"][-1]["content"] = f"{a1['name']}: {full1}"
                        st["convo"][-1]["content"] = full1
                        yield (st["history"], start_dis, stop_en, *opts_hidden, f"{a1['name']} Ÿ���� ��...")
                except Exception:
                    st["history"][-1]["content"] = f"{a1['name']}: (���� �� ����)"
                    st["convo"][-1]["content"] = "(���� �� ����)"

                # Agent 2
                full2 = ""
                st["history"].append({"role": "assistant", "content": f"{a2['name']}: "})
                st["convo"].append({"name": a2["name"], "content": ""})
                yield (st["history"], start_dis, stop_en, *opts_hidden, f"{a2['name']} Ÿ���� ��...")
                try:
                    for chunk in stream_generate(url, model, p2, options={"temperature": a2.get("temperature", 0.9), "top_k": a2.get("top_k", 40)}):
                        if not st.get("running"):
                            break
                        full2 += chunk
                        st["history"][-1]["content"] = f"{a2['name']}: {full2}"
                        st["convo"][-1]["content"] = full2
                        yield (st["history"], start_dis, stop_en, *opts_hidden, f"{a2['name']} Ÿ���� ��...")
                except Exception:
                    st["history"][-1]["content"] = f"{a2['name']}: (���� �� ����)"
                    st["convo"][-1]["content"] = "(���� �� ����)"

                yield (st["history"], start_dis, stop_en, *[gr.update(visible=True)]*5, "���� ������ �����ϼ���")
                return
            else:
                st["history"].append({"role": "user", "content": "새로운 질문을 입력하고 시작을 누르세요."})
                return (st["history"], gr.update(interactive=True), gr.update(interactive=True), *[gr.update(visible=False)]*5, "입력 대기")

            full = ""
            role = "user" if speaker["name"] == a1["name"] else "assistant"
            st["history"].append({"role": role, "content": f"{speaker['name']}: "})
            st["convo"].append({"name": speaker["name"], "content": ""})
            opts_hidden = [gr.update(visible=False)]*5
            # Immediate UI feedback before network call
            yield (
                st["history"],
                gr.update(interactive=False),
                gr.update(interactive=True),
                *opts_hidden,
                f"{speaker['name']} 준비 중...",
            )
            try:
                for chunk in stream_generate(url, model, prompt, options={"temperature": speaker.get("temperature", 0.95), "top_k": speaker.get("top_k", 40)}):
                    if not st.get("running"):
                        break
                    full += chunk
                    st["history"][-1]["content"] = f"{speaker['name']}: {full}"
                    st["convo"][-1]["content"] = full
                    yield (st["history"], gr.update(interactive=False), gr.update(interactive=True), *opts_hidden, f"{speaker['name']} 타이핑 중...")
            except Exception:
                st["history"][-1]["content"] = f"{speaker['name']}: (생성 중 오류)"
                st["convo"][-1]["content"] = "(생성 중 오류)"

            yield (st["history"], gr.update(interactive=False), gr.update(interactive=True), *[gr.update(visible=True)]*5, "다음 진행을 선택하세요")

        start_btn.click(
            on_start,
            inputs=[
                user_question, system_rules, ollama_url, model_dd, fast_mode,
                agent1_name, agent1_system, agent1_mem,
                agent2_name, agent2_system, agent2_mem,
                state,
            ],
            outputs=[chat, start_btn, stop_btn, opt_1, opt_2, opt_3, opt_4, opt_5, status],
            queue=True,
        )

        stop_btn.click(
            on_stop,
            inputs=[state],
            outputs=[chat, start_btn, stop_btn, opt_1, opt_2, opt_3, opt_4, opt_5, status],
        )

        reset_btn.click(
            on_reset,
            inputs=[state],
            outputs=[chat, start_btn, stop_btn, opt_1, opt_2, opt_3, opt_4, opt_5, status],
        )

        opt_1.click(
            proceed,
            inputs=[gr.State("opt1"), user_question, system_rules, ollama_url, model_dd, state],
            outputs=[chat, start_btn, stop_btn, opt_1, opt_2, opt_3, opt_4, opt_5, status],
            queue=True,
        )
        opt_2.click(
            proceed,
            inputs=[gr.State("opt2"), user_question, system_rules, ollama_url, model_dd, state],
            outputs=[chat, start_btn, stop_btn, opt_1, opt_2, opt_3, opt_4, opt_5, status],
            queue=True,
        )
        opt_3.click(
            proceed,
            inputs=[gr.State("opt3"), user_question, system_rules, ollama_url, model_dd, state],
            outputs=[chat, start_btn, stop_btn, opt_1, opt_2, opt_3, opt_4, opt_5, status],
            queue=True,
        )
        opt_4.click(
            proceed,
            inputs=[gr.State("opt4"), user_question, system_rules, ollama_url, model_dd, state],
            outputs=[chat, start_btn, stop_btn, opt_1, opt_2, opt_3, opt_4, opt_5, status],
            queue=True,
        )
        opt_5.click(
            proceed,
            inputs=[gr.State("opt5"), user_question, system_rules, ollama_url, model_dd, state],
            outputs=[chat, start_btn, stop_btn, opt_1, opt_2, opt_3, opt_4, opt_5, status],
            queue=True,
        )

    return demo


if __name__ == "__main__":
    app = main()
    # Use queue with defaults for compatibility across Gradio versions
    app.queue().launch()
