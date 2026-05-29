from __future__ import annotations

import argparse
import ast
import html
import json
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

SCENARIO_PATH = ROOT / "docs" / "design" / "agent" / "evaluation_scenario.md"
DEFAULT_REPORT_PATH = ROOT / "docs" / "design" / "agent" / "chenmo.html"
DEFAULT_JSON_PATH = ROOT / "docs" / "design" / "agent" / "chenmo_eval_results.json"

SCORING_RULES: dict[str, dict[str, list[str]]] = {
    "T1": {"must_any": ["2年2个月", "两年两个月", "2年", "两年"], "must_all": ["阿里", "字节"], "must_any_2": ["10个月", "十个月"], "must_any_3": ["阿里更长", "阿里时间更长"]},
    "T2": {"must_any": ["9个月", "九个月"]},
    "T3": {"must_all": ["2023年3月", "智云科技", "CTO"], "must_any": ["A轮", "融资困难", "融资不顺"]},
    "T4": {"must_all": ["甲状腺", "技术卓越奖", "Pre-A", "结婚"], "must_any": ["2020.8", "2021.3", "2022.1", "2022"]},
    "T5": {"must_any": ["已完成", "完成了"], "must_all": ["B轮", "2024年9月"]},
    "T6": {"must_all": ["字节", "智云", "2022年6月"], "must_any": ["筹备期", "成立后产品开发期", "成立后开发期"]},
    "M1": {"must_all": ["云梯科技", "钉钉", "阿里"]},
    "M2": {"must_all": ["张薇", "前女友", "金沙江"]},
    "M3": {"must_all": ["百川智能", "大模型"], "must_any": ["竞对", "竞争", "转型方向一致"]},
    "M4": {"must_all": ["小红书", "字节"]},
    "M5": {"must_all": ["阿里", "陈默", "联合创始人"]},
    "M6": {"must_any": ["Google Brain", "谷歌"], "must_all": ["小米汽车"], "must_any_2": ["没有直接关系", "独立公司", "科技公司"]},
    "M7": {"must_all": ["小米", "小红书"], "must_any": ["不是同一家公司", "不是同一"]},
    "C1": {"must_any": ["P2-2", "60%的薪资涨幅", "60%涨幅"], "must_all": ["健康", "甲状腺"], "must_any_2": ["减少压力", "疫情期间加班"]},
    "C2": {"must_all": ["战略分歧", "大模型", "先做收入"], "must_any": ["吵架", "意见不合", "离职"]},
    "C3": {"must_all": ["现金流断裂", "裁员"], "must_any": ["救公司", "维持运营", "注资"], "must_any_2": ["马东", "抵押房子"]},
    "C4": {"must_any": ["Llama 2", "大模型"], "must_all": ["2024年2月", "新版本上线"], "must_any_2": ["营收", "续约率提升", "竞争力增强"]},
    "C5": {"must_all": ["父亲", "远程管理", "产品团队"], "must_any": ["吃力", "拆分", "组织架构调整"]},
    "C6": {"must_all": ["C轮融资", "独角兽"], "must_any": ["关键期", "不能离开", "核心高管"]},
    "C7": {"must_all": ["抵押", "房子"], "must_any": ["责任感", "担当", "CEO"]},
    "CT1": {"must_all": ["陈默", "智云科技", "云梯科技", "钉钉"]},
    "CT2": {"must_all": ["技术卓越奖", "协同编辑", "冲突解决"], "must_any": ["能力迁移", "产品竞争力", "B轮融资"]},
    "CT3": {"must_all": ["CTO", "产品VP"], "must_any": ["丈夫", "父亲", "儿子"], "must_any_2": ["家庭责任", "工作冲突", "依赖"]},
    "CT4": {"must_all": ["周强", "张薇", "金沙江", "B轮"], "must_any": ["裁员", "注资", "Llama 2", "大模型"]},
    "CT5": {"must_any": ["健康风险", "更健康的工作环境"], "must_any_2": ["CTO转产品VP", "产品VP"], "must_any_3": ["双重身份", "期权", "财务安全垫"]},
    "CT6": {"must_all": ["父亲", "远程管理120人", "王磊"], "must_any": ["儿子", "C轮", "拆分产品团队", "国内线", "国际线"]},
    "COMP1": {"must_all": ["字节"], "must_any": ["不会经历创业危机", "不会经历2023年的创业生死危机"], "must_any_2": ["张薇", "财务自由", "独角兽"]},
    "COMP2": {"must_all": ["马东", "张薇", "李婷"], "must_any": ["创业机会", "牵线", "精神支持", "家庭责任"]},
    "COMP3": {"must_all": ["阿里", "字节", "智云"], "must_any": ["忽视健康", "重视健康", "高压", "家庭优先"]},
    "COMP4": {"must_all": ["40%", "15%", "10%", "35%", "金沙江", "高瓴"], "must_any": ["腾讯", "董事会多元化", "股份稀释"]},
    "COMP5": {"must_all": ["C轮", "独角兽", "月供"], "must_any": ["股权未变现", "IPO", "不确定性", "财务自由"]},
}


@dataclass
class Turn:
    role: str
    content: str
    timestamp: int


@dataclass
class Question:
    section: str
    question_id: str
    prompt: str
    expected: str
    reasoning: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Chenmo long-horizon graph-memory evaluation.")
    parser.add_argument("--scenario", default=str(SCENARIO_PATH))
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--json", default=str(DEFAULT_JSON_PATH))
    parser.add_argument("--agent-url", default="http://127.0.0.1:8765")
    parser.add_argument("--memory-url", default="http://127.0.0.1:8000")
    parser.add_argument("--config", default=None)
    parser.add_argument("--commit-timeout-seconds", type=int, default=1800)
    parser.add_argument(
        "--progress-jsonl",
        default=str(ROOT / "docs" / "design" / "agent" / "chenmo_eval_progress.jsonl"),
        help="Path to newline-delimited JSON progress events.",
    )
    parser.add_argument(
        "--partial-json",
        default=str(ROOT / "docs" / "design" / "agent" / "chenmo_eval_partial.json"),
        help="Path to partial JSON snapshot updated after each QA.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario_path = Path(args.scenario)
    report_path = Path(args.report)
    json_path = Path(args.json)
    progress_path = Path(args.progress_jsonl)
    partial_json_path = Path(args.partial_json)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    partial_json_path.parent.mkdir(parents=True, exist_ok=True)
    if progress_path.exists():
        progress_path.unlink()

    scenario = parse_scenario(scenario_path)
    questions = parse_questions(scenario_path)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    label = f"chenmo-eval-{stamp}"
    account = create_account(args.agent_url, label)
    account_info = account["account"]
    auth_key = str(account_info["authKey"])
    tenant_id = str(account_info["tenantId"])
    user_id = str(account_info["userId"])
    ingest_session_id = f"chenmo-ingest-{stamp}"
    qa_session_prefix = f"chenmo-qa-{stamp}"
    agent_id = "chenmo-eval-agent"

    stage_event(
        progress_path,
        "bootstrap",
        "running",
        {
            "scenario_path": str(scenario_path),
            "question_count": len(questions),
            "label": label,
        },
    )
    turns = build_turns(scenario)
    stage_event(
        progress_path,
        "import",
        "running",
        {"session_id": ingest_session_id, "turn_count": len(turns)},
    )
    open_session(args.memory_url, auth_key, user_id, agent_id, ingest_session_id)
    for index, turn in enumerate(turns):
        add_message(
            args.memory_url,
            auth_key,
            ingest_session_id,
            {
                "role": turn.role,
                "content": turn.content,
                "metadata": {
                    "evaluation_case": "chenmo",
                    "turn_index": index,
                    "event_time": turn.timestamp,
                    "event_datetime": datetime.fromtimestamp(turn.timestamp, UTC).isoformat(),
                },
            },
        )
        if (index + 1) % 10 == 0 or (index + 1) == len(turns):
            stage_event(
                progress_path,
                "import",
                "progress",
                {"imported_turns": index + 1, "total_turns": len(turns)},
            )
    commit = post_json(
        f"{args.memory_url}/api/sessions/{ingest_session_id}/commit",
        {"metadata": {"evaluation_case": "chenmo", "label": label}},
        headers={"X-Auth-Key": auth_key},
    )
    commit_id = str((commit.get("result") or {}).get("commit_id") or (commit.get("result") or {}).get("archive_id") or "")
    commit_status = wait_commit(
        args.memory_url,
        auth_key,
        ingest_session_id,
        commit_id,
        timeout_seconds=max(60, int(args.commit_timeout_seconds)),
    )
    stage_event(
        progress_path,
        "import",
        "completed",
        {"session_id": ingest_session_id, "commit_id": commit_id, "commit_status": commit_status},
    )

    results = []
    stage_event(
        progress_path,
        "qa",
        "running",
        {"total_questions": len(questions)},
    )
    for index, question in enumerate(questions):
        qa_number = index + 1
        qa_session_id = f"{qa_session_prefix}-{index:02d}-{question.question_id.lower()}"
        stage_event(
            progress_path,
            "qa_item",
            "running",
            {"number": qa_number, "question_id": question.question_id, "question": question.prompt},
        )
        retrieval = search_retrieval(
            args.memory_url,
            auth_key,
            question.prompt,
            agent_id,
            qa_session_id,
            limit=8,
        )
        answer = ask_agent(
            args.agent_url,
            auth_key,
            user_id,
            agent_id,
            qa_session_id,
            question.prompt,
        )
        judgment = judge_answer(question, answer["assistant"]["content"])
        retrieval_failure = classify_retrieval(retrieval)
        results.append(
            {
                "number": qa_number,
                "section": question.section,
                "question_id": question.question_id,
                "question": question.prompt,
                "expected": question.expected,
                "reasoning": question.reasoning,
                "answer": answer["assistant"]["content"],
                "retrieval": retrieval,
                "judgment": judgment,
                "retrieval_failure": retrieval_failure,
            }
        )
        partial_report = {
            "generated_at": datetime.now(UTC).isoformat(),
            "scenario_path": str(scenario_path),
            "tenant_id": tenant_id,
            "user_id": user_id,
            "ingest_session_id": ingest_session_id,
            "commit_id": commit_id,
            "commit_status": commit_status,
            "turn_count": len(turns),
            "question_count": len(questions),
            "completed_questions": qa_number,
            "summary": build_summary(results),
            "analysis": build_failure_analysis(results),
            "results": results,
        }
        partial_json_path.write_text(json.dumps(partial_report, ensure_ascii=False, indent=2), encoding="utf-8")
        stage_event(
            progress_path,
            "qa_item",
            "completed",
            {
                "number": qa_number,
                "question_id": question.question_id,
                "passed": bool(judgment.get("passed")),
                "score": float(judgment.get("score") or 0.0),
                "retrieval_failure": retrieval_failure,
            },
        )

    summary = build_summary(results)
    analysis = build_failure_analysis(results)
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "scenario_path": str(scenario_path),
        "report_path": str(report_path),
        "json_path": str(json_path),
        "tenant_id": tenant_id,
        "user_id": user_id,
        "ingest_session_id": ingest_session_id,
        "commit_id": commit_id,
        "commit_status": commit_status,
        "turn_count": len(turns),
        "question_count": len(questions),
        "summary": summary,
        "analysis": analysis,
        "results": results,
    }

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(render_html(report), encoding="utf-8")
    stage_event(
        progress_path,
        "qa",
        "completed",
        {
            "total": summary.get("total"),
            "passed": summary.get("passed"),
            "failed": summary.get("failed"),
            "pass_rate": summary.get("pass_rate"),
            "report": str(report_path),
            "json": str(json_path),
        },
    )
    print(json.dumps({"report": str(report_path), "json": str(json_path), "summary": summary}, ensure_ascii=False, indent=2))


def stage_event(path: Path, stage: str, status: str, payload: dict[str, Any]) -> None:
    event = {
        "ts": datetime.now(UTC).isoformat(),
        "stage": stage,
        "status": status,
        "payload": payload,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    print(json.dumps(event, ensure_ascii=False))


def parse_scenario(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S)
    if not match:
        raise ValueError(f"scenario JSON block not found in {path}")
    payload = ast.literal_eval(match.group(1))
    if not isinstance(payload, dict):
        raise ValueError("scenario payload must be a dict")
    return payload


def parse_questions(path: Path) -> list[Question]:
    questions: list[Question] = []
    section = ""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("### "):
            section = line[4:].strip()
            continue
        if not line.startswith("| ") or line.startswith("|------") or "答案" in line and "推理要点" in line:
            continue
        cols = [col.strip() for col in line.strip("|").split("|")]
        if len(cols) != 4:
            continue
        question_id, prompt, expected, reasoning = cols
        if not re.match(r"^[A-Z]+[0-9]+$", question_id):
            continue
        questions.append(
            Question(
                section=section,
                question_id=question_id,
                prompt=prompt,
                expected=expected,
                reasoning=reasoning,
            )
        )
    return questions


def build_turns(scenario: dict[str, Any]) -> list[Turn]:
    turns_data = scenario.get("turns")
    if not isinstance(turns_data, list):
        raise ValueError("scenario turns must be a list")
    default_time = datetime(2018, 1, 1, tzinfo=UTC)
    turns: list[Turn] = []
    for index, item in enumerate(turns_data):
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError(f"invalid turn at index {index}: {item!r}")
        role, content = item
        timestamp = infer_timestamp(str(content), default_time, index)
        default_time = datetime.fromtimestamp(timestamp, UTC) + timedelta(minutes=1)
        turns.append(Turn(role=str(role), content=str(content), timestamp=timestamp))
    return turns


def infer_timestamp(content: str, fallback: datetime, index: int) -> int:
    patterns = [
        r"(?P<y>\d{4})年(?P<m>\d{1,2})月(?P<d>\d{1,2})日",
        r"(?P<y>\d{4})\.(?P<m>\d{1,2})",
        r"(?P<y>\d{4})年(?P<m>\d{1,2})月",
    ]
    for pattern in patterns:
        match = re.search(pattern, content)
        if not match:
            continue
        year = int(match.group("y"))
        month = int(match.group("m"))
        day = int(match.groupdict().get("d") or 1)
        dt = datetime(year, month, day, tzinfo=UTC) + timedelta(minutes=index)
        return int(dt.timestamp())
    return int((fallback + timedelta(minutes=index)).timestamp())


def create_account(agent_url: str, label: str) -> dict[str, Any]:
    return post_json(f"{agent_url}/agent/accounts/create", {"label": label})


def open_session(memory_url: str, auth_key: str, user_id: str, agent_id: str, session_id: str) -> dict[str, Any]:
    return post_json(
        f"{memory_url}/api/sessions/open",
        {"agent_id": agent_id, "session_id": session_id},
        headers={"X-Auth-Key": auth_key},
    )


def add_message(memory_url: str, auth_key: str, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    return post_json(
        f"{memory_url}/api/sessions/{session_id}/messages",
        payload,
        headers={"X-Auth-Key": auth_key},
    )


def wait_commit(
    memory_url: str,
    auth_key: str,
    session_id: str,
    commit_id: str,
    *,
    timeout_seconds: int = 600,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_status: dict[str, Any] = {}
    while time.monotonic() < deadline:
        data = get_json(
            f"{memory_url}/api/sessions/{session_id}/commits/{commit_id}",
            headers={"X-Auth-Key": auth_key},
        )
        status = data.get("status") if isinstance(data.get("status"), dict) else {}
        last_status = dict(status)
        value = str(status.get("status") or "")
        if value == "completed":
            return last_status
        if value == "failed":
            raise RuntimeError(f"commit failed: {status}")
        time.sleep(2)
    raise TimeoutError(f"commit timeout: {commit_id}; last_status={last_status}")


def search_retrieval(
    memory_url: str,
    auth_key: str,
    query: str,
    agent_id: str,
    session_id: str,
    *,
    limit: int,
) -> dict[str, Any]:
    return post_json(
        f"{memory_url}/api/retrieval/search",
        {
            "query": query,
            "agent_id": agent_id,
            "session_id": session_id,
            "limit": limit,
            "include_explain": True,
        },
        headers={"X-Auth-Key": auth_key},
    )


def ask_agent(
    agent_url: str,
    auth_key: str,
    user_id: str,
    agent_id: str,
    session_id: str,
    message: str,
) -> dict[str, Any]:
    return post_json(
        f"{agent_url}/agent/chat",
        {
            "user_id": user_id,
            "agent_id": agent_id,
            "session_id": session_id,
            "message": message,
            "include_history": False,
        },
        headers={"X-Auth-Key": auth_key},
    )


def judge_answer(question: Question, answer: str) -> dict[str, Any]:
    rules = SCORING_RULES.get(question.question_id, {})
    misses: list[str] = []
    hits = 0
    checks = 0
    for key, terms in rules.items():
        checks += 1
        if key.startswith("must_all"):
            matched = all(term in answer for term in terms)
        else:
            matched = any(term in answer for term in terms)
        if matched:
            hits += 1
        else:
            misses.append(f"{key}:{'/'.join(terms)}")
    score = round(hits / checks, 4) if checks else 0.0
    contradictions = []
    if question.question_id == "T5" and "没完成" in answer:
        contradictions.append("说成 B 轮未完成")
    if question.question_id == "M7" and "同一家公司" in answer:
        contradictions.append("误判为同一家公司")
    if question.question_id == "C1" and "只有薪资原因" in answer:
        contradictions.append("遗漏健康根因")
    passed = score >= 0.75 and not contradictions
    if question.question_id in {"COMP1", "COMP2", "COMP3", "COMP4", "COMP5", "CT2", "CT3", "CT4", "CT5", "CT6"}:
        passed = score >= 0.67 and not contradictions
    return {
        "passed": passed,
        "score": score,
        "reason": "规则命中充分" if passed else "规则命中不足",
        "missing_points": misses,
        "contradictions": contradictions,
        "raw": "",
    }


def classify_retrieval(retrieval: dict[str, Any]) -> str:
    result = retrieval.get("result") if isinstance(retrieval.get("result"), dict) else retrieval
    items = result.get("items") if isinstance(result, dict) else []
    if not items:
        return "empty"
    relation_count = sum(1 for item in items if str(item.get("kind") or "") == "relation")
    episode_count = sum(1 for item in items if str(item.get("kind") or "") == "episode")
    if relation_count == 0 and episode_count > 0:
        return "episodes_only"
    if relation_count == 0:
        return "non_graph_only"
    return "ok"


def build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_section: dict[str, dict[str, Any]] = {}
    total_passed = 0
    for item in results:
        section = item["section"]
        section_bucket = by_section.setdefault(section, {"total": 0, "passed": 0, "avg_score": 0.0})
        section_bucket["total"] += 1
        score = float(item["judgment"].get("score") or 0.0)
        section_bucket["avg_score"] += score
        if item["judgment"].get("passed"):
            total_passed += 1
            section_bucket["passed"] += 1
    for value in by_section.values():
        total = value["total"] or 1
        value["pass_rate"] = round(value["passed"] / total, 4)
        value["avg_score"] = round(value["avg_score"] / total, 4)
    return {
        "total": len(results),
        "passed": total_passed,
        "failed": len(results) - total_passed,
        "pass_rate": round(total_passed / len(results), 4) if results else 0.0,
        "by_section": by_section,
    }


def build_failure_analysis(results: list[dict[str, Any]]) -> dict[str, Any]:
    failures = [item for item in results if not item["judgment"].get("passed")]
    categories: Counter[str] = Counter()
    section_breakdown: dict[str, int] = Counter()
    top_examples: list[dict[str, Any]] = []

    for item in failures:
        section_breakdown[item["section"]] += 1
        if item["retrieval_failure"] == "empty":
            categories["retrieval_empty"] += 1
        elif item["retrieval_failure"] in {"episodes_only", "non_graph_only"}:
            categories["retrieval_understructured"] += 1
        if item["judgment"].get("contradictions"):
            categories["answer_contradiction"] += 1
        if item["section"].startswith("一、时序"):
            categories["temporal_reasoning_gap"] += 1
        elif item["section"].startswith("二、多跳"):
            categories["multi_hop_reasoning_gap"] += 1
        elif item["section"].startswith("三、因果"):
            categories["causal_reasoning_gap"] += 1
        elif item["section"].startswith("四、复杂任务"):
            categories["task_reasoning_gap"] += 1
        elif item["section"].startswith("五、综合"):
            categories["comprehensive_reasoning_gap"] += 1
        top_examples.append(
            {
                "question_id": item["question_id"],
                "section": item["section"],
                "question": item["question"],
                "reason": item["judgment"].get("reason"),
                "retrieval_failure": item["retrieval_failure"],
            }
        )

    recommendations = [
        "抽取层从“扁平三元组”升级为“Episode + Entity + Relation + Event + Task + Claim”六层模型，把任职、融资、婚姻、手术、邀请、拒绝、牵线、转型分歧等显式建模为事件，并给事件挂时间、参与方、结果、证据。",
        "按 Graphiti 的思路强化时间和溯源：每条事实保留 validity window、source episode、evidence span，支持查询“当时为真”而不是只查当前真值。Graphiti README 重点强调了 validity windows、episodes provenance 和 hybrid retrieval。",
        "把检索前的意图识别升级为“问题分解器”：先判断是时序、多跳、因果、任务、反事实还是风险评估，再把一个问题拆成多个 graph sub-queries，例如先找人物/公司，再追事件，再做时间过滤和路径拼接。",
        "增加图检索后的二阶段推理：先返回候选子图，再由 answer planner 基于子图做比较、排序、因果链串联，避免仅凭单跳 relation 直接生成答案。",
        "为 task 建立一级对象，而不是把任务散落在 episode 文本里。任务应有 task_id、状态、目标、依赖、参与人、相关地点/文档/客户，才能稳定回答“关于这次出行/这个项目/这轮融资聊了什么”。",
        "给人物和公司增加 evolving summary / timeline snapshot，尤其是长期实体如陈默、马东、智云科技、李婷。复杂题往往需要跨 3-8 个 episode 汇总，没有实体时间线就会退化成全文回忆。",
        "新增多跳检索策略：先图遍历找到 1-hop/2-hop/3-hop 子图，再做 BM25/embedding 混排和 graph-distance rerank。Graphiti README 把 hybrid retrieval 和 graph-distance rerank 作为关键能力。",
        "因果题单独增加 CauseEdge / DecisionEdge / ConstraintEdge 抽取，让“健康问题导致跳槽”“父亲手术导致组织拆分建议”不再只能从原文隐式猜。",
        "综合题增加长期目标与风险状态图谱，例如 Goal(2028 IPO)、Risk(C轮未落地、家庭两地压力、月供压力)、Mitigation(组织拆分、双重角色)，这样评估题能直接检索目标-进展-风险三段证据。",
        "评测侧保留可回放 artifacts：每题的 retrieval 命中、answer、judge JSON、LLM 提示词和 graph subgraph 截图，形成稳定 regression harness，而不是只看最终一句回答。",
    ]

    return {
        "failure_count": len(failures),
        "failure_sections": dict(section_breakdown),
        "failure_categories": dict(categories),
        "top_examples": top_examples[:12],
        "recommendations": recommendations,
    }


def render_html(report: dict[str, Any]) -> str:
    summary = report["summary"]
    analysis = report["analysis"]
    rows = []
    for item in report["results"]:
        judgment = item["judgment"]
        retrieval_items = extract_texts(item["retrieval"])
        rows.append(
            f"""
            <tr class="{'pass' if judgment.get('passed') else 'fail'}">
              <td>{html.escape(item['question_id'])}</td>
              <td>{html.escape(item['section'])}</td>
              <td>{html.escape(item['question'])}</td>
              <td>{'通过' if judgment.get('passed') else '未通过'}</td>
              <td>{float(judgment.get('score') or 0.0):.2f}</td>
              <td>{html.escape(item['answer'])}</td>
              <td>{html.escape(item['expected'])}</td>
              <td>{html.escape(judgment.get('reason') or '')}</td>
              <td>{html.escape(item['retrieval_failure'])}</td>
              <td>{html.escape(' | '.join(retrieval_items[:6]))}</td>
            </tr>
            """
        )

    section_cards = []
    for name, value in summary["by_section"].items():
        section_cards.append(
            f"""
            <div class="card">
              <h3>{html.escape(name)}</h3>
              <p>通过 {value['passed']} / {value['total']}</p>
              <p>通过率 {value['pass_rate']:.0%}</p>
              <p>平均分 {value['avg_score']:.2f}</p>
            </div>
            """
        )

    failure_examples = []
    for example in analysis["top_examples"]:
        failure_examples.append(
            f"<li><strong>{html.escape(example['question_id'])}</strong> {html.escape(example['question'])} | {html.escape(example['reason'] or '')} | retrieval={html.escape(example['retrieval_failure'])}</li>"
        )

    failure_categories = []
    for key, value in analysis["failure_categories"].items():
        failure_categories.append(f"<li>{html.escape(key)}: {value}</li>")

    recommendations = []
    for item in analysis["recommendations"]:
        recommendations.append(f"<li>{html.escape(item)}</li>")

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>陈默场景评测报告</title>
  <style>
    :root {{
      --bg: #f6f3ea;
      --paper: #fffdf8;
      --ink: #1f2430;
      --muted: #6d7380;
      --line: #ddd4c4;
      --accent: #0f5d66;
      --good: #196b47;
      --bad: #a64035;
      --warn: #946200;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background:
        radial-gradient(circle at top right, rgba(15,93,102,0.12), transparent 28%),
        linear-gradient(180deg, #f5efe1 0%, var(--bg) 42%, #f8f5ee 100%);
      color: var(--ink);
      font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      line-height: 1.6;
    }}
    .wrap {{ max-width: 1440px; margin: 0 auto; padding: 28px; }}
    .hero {{
      padding: 28px;
      border: 1px solid rgba(31,36,48,0.08);
      border-radius: 24px;
      background: linear-gradient(145deg, rgba(255,255,255,0.92), rgba(255,251,242,0.98));
      box-shadow: 0 20px 60px rgba(31,36,48,0.08);
    }}
    .eyebrow {{
      display: inline-block;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(15,93,102,0.10);
      color: var(--accent);
      font-size: 12px;
      font-weight: 800;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }}
    h1 {{ margin: 12px 0 8px; font-size: 34px; line-height: 1.15; }}
    h2 {{ margin: 0 0 12px; font-size: 22px; }}
    p {{ margin: 8px 0; color: var(--muted); }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
      margin-top: 22px;
    }}
    .stat, .card, .panel {{
      padding: 18px;
      border: 1px solid var(--line);
      border-radius: 18px;
      background: var(--paper);
    }}
    .stat strong {{ display: block; font-size: 30px; color: var(--accent); }}
    .grid {{
      display: grid;
      grid-template-columns: 1.1fr 0.9fr;
      gap: 18px;
      margin-top: 18px;
    }}
    .card-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}
    ul {{ margin: 10px 0 0; padding-left: 20px; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 14px;
      font-size: 13px;
      background: var(--paper);
    }}
    th, td {{
      padding: 10px;
      border: 1px solid var(--line);
      vertical-align: top;
      text-align: left;
    }}
    th {{
      position: sticky;
      top: 0;
      background: #efe6d4;
      z-index: 1;
    }}
    tr.pass td:first-child {{ box-shadow: inset 4px 0 0 var(--good); }}
    tr.fail td:first-child {{ box-shadow: inset 4px 0 0 var(--bad); }}
    .section {{
      margin-top: 18px;
      padding: 22px;
      border: 1px solid var(--line);
      border-radius: 22px;
      background: rgba(255,253,248,0.95);
    }}
    .mono {{ font-family: "Cascadia Code", Consolas, monospace; font-size: 12px; color: var(--ink); }}
    .kpis {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin-top: 12px;
    }}
    .good {{ color: var(--good); }}
    .bad {{ color: var(--bad); }}
    .warn {{ color: var(--warn); }}
    @media (max-width: 1100px) {{
      .stats, .grid, .card-grid, .kpis {{ grid-template-columns: 1fr; }}
      .wrap {{ padding: 16px; }}
      table {{ display: block; overflow: auto; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <span class="eyebrow">Chenmo Evaluation</span>
      <h1>陈默长时记忆评测报告</h1>
      <p>基于 <span class="mono">{html.escape(report['scenario_path'])}</span> 真实回放 55 条对话，完成 commit 后对 30 个问题逐题发问、逐题判分，并输出失败归因和重构建议。</p>
      <div class="stats">
        <div class="stat"><strong>{summary['passed']}/{summary['total']}</strong><span>通过题数</span></div>
        <div class="stat"><strong>{summary['pass_rate']:.0%}</strong><span>总通过率</span></div>
        <div class="stat"><strong>{report['turn_count']}</strong><span>导入对话轮数</span></div>
        <div class="stat"><strong>{analysis['failure_count']}</strong><span>失败题数</span></div>
      </div>
      <div class="kpis">
        <div class="panel">
          <h2>运行信息</h2>
          <p>tenant: <span class="mono">{html.escape(report['tenant_id'])}</span></p>
          <p>user: <span class="mono">{html.escape(report['user_id'])}</span></p>
          <p>ingest session: <span class="mono">{html.escape(report['ingest_session_id'])}</span></p>
          <p>commit: <span class="mono">{html.escape(report['commit_id'])}</span></p>
          <p>generated at: <span class="mono">{html.escape(report['generated_at'])}</span></p>
        </div>
        <div class="panel">
          <h2>结论</h2>
          <p>当前实现已经能覆盖一部分单跳和部分时序题，但在多跳、复杂因果、任务堆叠和综合评估类题上仍有明显短板。</p>
          <p>失败主因不是单点 bug，而是图模式、检索计划和 answer planner 都还偏“局部事实召回”，不足以稳定支撑长程子图推理。</p>
        </div>
        <div class="panel">
          <h2>Graphiti 启发</h2>
          <p>Graphiti 强调 temporal validity windows、episode provenance、prescribed/learned ontology，以及 hybrid retrieval（semantic + keyword + graph traversal）。这些点和本场景失败模式高度对齐。</p>
          <p class="mono">Sources: https://github.com/getzep/graphiti</p>
        </div>
      </div>
    </section>

    <section class="section">
      <h2>分维度成绩</h2>
      <div class="card-grid">
        {''.join(section_cards)}
      </div>
    </section>

    <section class="section">
      <div class="grid">
        <div class="panel">
          <h2>失败归因</h2>
          <ul>
            {''.join(failure_categories)}
          </ul>
        </div>
        <div class="panel">
          <h2>失败样例</h2>
          <ul>
            {''.join(failure_examples)}
          </ul>
        </div>
      </div>
    </section>

    <section class="section">
      <h2>建议方案</h2>
      <ul>
        {''.join(recommendations)}
      </ul>
    </section>

    <section class="section">
      <h2>逐题结果</h2>
      <table>
        <thead>
          <tr>
            <th>ID</th>
            <th>维度</th>
            <th>问题</th>
            <th>结果</th>
            <th>分数</th>
            <th>系统回答</th>
            <th>标准答案</th>
            <th>判分原因</th>
            <th>检索态</th>
            <th>检索证据</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows)}
        </tbody>
      </table>
    </section>
  </div>
</body>
</html>
"""


def extract_texts(retrieval: dict[str, Any]) -> list[str]:
    result = retrieval.get("result") if isinstance(retrieval.get("result"), dict) else retrieval
    items = result.get("items") if isinstance(result, dict) else []
    texts = []
    for item in items:
        text = str(item.get("text") or item.get("content") or "").strip()
        if text:
            texts.append(text)
    return texts


def get_json(url: str, headers: dict[str, str] | None = None) -> dict[str, Any]:
    request = Request(url, headers=headers or {}, method="GET")
    return read_json(request)


def post_json(url: str, payload: dict[str, Any], headers: dict[str, str] | None = None) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8", **(headers or {})},
        method="POST",
    )
    return read_json(request)


def read_json(request: Request) -> dict[str, Any]:
    try:
        with urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"http_{exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"url_error: {exc.reason}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"response is not a JSON object: {data!r}")
    return data


if __name__ == "__main__":
    main()
