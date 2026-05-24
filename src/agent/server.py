"""Standalone agent playground server.

This package intentionally avoids importing echomem internals. It can be moved
to a separate project and continue to work against the EchoMemory HTTP API.
"""

from __future__ import annotations

import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from .chat_service import AgentChatService
from .config import AgentConfig, load_config
from .echomemory_client import EchoMemoryClientError
from .model_client import ModelClientError
INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>EchoMemory 智能体对话</title>
  <style>
    :root {
      --ink: #182235;
      --muted: #66758b;
      --line: #d7e0ec;
      --paper: #fbfcff;
      --blue: #2357c6;
      --green: #16734d;
      --orange: #a65a16;
      --red: #a23a3a;
      --soft: #f6f7fb;
      --sidebar: #f3f4f8;
      --brand: #5f6df4;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      background: #eef1f6;
      font-family: "Segoe UI", "Microsoft YaHei", Arial, sans-serif;
      letter-spacing: 0;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 14px 18px;
      border-bottom: 1px solid var(--line);
      background: #fff;
    }
    h1 { margin: 0; font-size: 20px; }
    .status { color: var(--muted); font-size: 13px; }
    main {
      display: grid;
      grid-template-columns: minmax(620px, 1.22fr) minmax(360px, 0.78fr);
      gap: 14px;
      padding: 14px;
      min-height: calc(100vh - 58px);
    }
    section {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--paper);
      overflow: hidden;
    }
    .panel-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      background: #fff;
    }
    h2 { margin: 0; font-size: 16px; }
    .panel-body { padding: 14px; }
    .chat-section {
      border: 0;
      border-radius: 14px;
      background: #fff;
      box-shadow: 0 14px 40px rgba(36, 45, 74, 0.08);
    }
    .chat-shell {
      display: grid;
      grid-template-columns: 232px minmax(0, 1fr);
      height: calc(100vh - 86px);
      min-height: 660px;
      background: #fff;
    }
    .chat-nav {
      display: flex;
      flex-direction: column;
      gap: 12px;
      padding: 14px 12px;
      border-right: 1px solid #eceef5;
      background: var(--sidebar);
    }
    .brand-row {
      display: flex;
      align-items: center;
      gap: 9px;
      min-height: 38px;
      padding: 2px 4px;
      font-weight: 800;
    }
    .brand-dot {
      display: grid;
      place-items: center;
      width: 30px;
      height: 30px;
      border-radius: 9px;
      background: #111827;
      color: #fff;
      font-size: 14px;
    }
    .nav-button {
      display: flex;
      align-items: center;
      gap: 8px;
      width: 100%;
      min-height: 38px;
      padding: 8px 10px;
      border: 0;
      border-radius: 10px;
      background: #fff;
      box-shadow: 0 6px 18px rgba(46, 55, 88, 0.06);
    }
    .nav-button.primary { background: #e9ecff; color: #3141d0; }
    .nav-list {
      display: grid;
      gap: 4px;
      margin-top: 4px;
    }
    .nav-item {
      display: flex;
      align-items: center;
      gap: 9px;
      min-height: 34px;
      padding: 7px 10px;
      border: 0;
      border-radius: 9px;
      background: transparent;
      color: #394357;
      font-size: 13px;
      font-weight: 700;
      text-align: left;
    }
    .nav-item.active, .nav-item:hover { background: #e7eaf3; }
    .nav-section-title {
      margin: 10px 8px 2px;
      color: #8a93a6;
      font-size: 12px;
      font-weight: 800;
    }
    .recent-chat {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 26px;
      gap: 6px;
      align-items: center;
      min-height: 32px;
      padding: 4px 5px 4px 10px;
      border-radius: 9px;
      color: #5c6577;
      font-size: 13px;
      cursor: pointer;
    }
    .recent-chat.active { background: #fff; color: #182235; font-weight: 800; }
    .recent-title {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .delete-session {
      display: grid;
      place-items: center;
      width: 24px;
      height: 24px;
      min-height: 24px;
      padding: 0;
      border: 0;
      border-radius: 7px;
      background: transparent;
      color: #9aa3b5;
      font-size: 16px;
    }
    .delete-session:hover { background: #eef0f6; color: #a23a3a; }
    .chat-nav-footer {
      margin-top: auto;
      padding: 10px;
      border-radius: 12px;
      background: #fff;
      color: #596274;
      font-size: 12px;
      line-height: 1.5;
    }
    .chat-main {
      display: grid;
      grid-template-rows: minmax(0, 1fr);
      height: 100%;
      min-height: 0;
      min-width: 0;
      background: linear-gradient(180deg, #fbfcff 0%, #fff 32%);
    }
    .main-view {
      display: grid;
      grid-template-rows: auto 1fr auto;
      height: 100%;
      min-height: 0;
      overflow: hidden;
    }
    .main-view.hidden { display: none; }
    .memory-main {
      display: none;
      grid-template-rows: auto 1fr;
      min-width: 0;
      background: #fbfcff;
    }
    .memory-main.active { display: grid; }
    .memory-page {
      min-height: 0;
      overflow: auto;
      padding: 22px;
    }
    .memory-hero {
      margin-bottom: 16px;
      padding: 18px;
      border: 1px solid #e2e7f0;
      border-radius: 14px;
      background: #fff;
      box-shadow: 0 8px 28px rgba(43, 50, 77, 0.06);
    }
    .memory-hero h2 {
      margin: 0 0 8px;
      font-size: 24px;
    }
    .memory-hero p {
      margin: 0;
      color: #66758b;
      line-height: 1.6;
    }
    .memory-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }
    .memory-card {
      padding: 14px;
      border: 1px solid #e2e7f0;
      border-radius: 12px;
      background: #fff;
    }
    .memory-card strong {
      display: block;
      margin-bottom: 6px;
      color: #25314a;
      font-size: 13px;
    }
    .chat-topbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      min-height: 58px;
      padding: 11px 18px;
      border-bottom: 1px solid #eff1f6;
      background: rgba(255, 255, 255, 0.92);
    }
    .chat-title {
      display: flex;
      align-items: center;
      gap: 8px;
      min-width: 0;
      font-size: 15px;
      font-weight: 850;
    }
    .model-pill {
      max-width: 230px;
      padding: 6px 9px;
      border-radius: 999px;
      background: #f2f4ff;
      color: #4a55d7;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-size: 12px;
      font-weight: 800;
    }
    .identity-row {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      padding: 10px 18px 0;
    }
    .identity-row label { color: #7a8495; }
    .identity-row input {
      min-height: 34px;
      border-color: #e4e7ef;
      border-radius: 10px;
      background: #fafbfe;
    }
    .chat-scroll {
      display: flex;
      flex-direction: column;
      min-height: 0;
      padding: 8px 0 0;
      overflow: auto;
    }
    .welcome {
      display: grid;
      place-items: center;
      min-height: 168px;
      padding: 20px;
      text-align: center;
    }
    .welcome h2 {
      margin: 0 0 8px;
      font-size: 28px;
      line-height: 1.2;
    }
    .welcome p {
      margin: 0;
      color: #6c7587;
      font-size: 14px;
    }
    .quick-prompts {
      display: flex;
      flex-wrap: wrap;
      justify-content: center;
      gap: 8px;
      margin-top: 16px;
    }
    .quick-prompts button {
      min-height: 32px;
      border: 1px solid #e5e8f1;
      border-radius: 999px;
      background: #fff;
      color: #4a5367;
      font-size: 13px;
      font-weight: 700;
    }
    .row {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 12px;
    }
    label { display: block; color: var(--muted); font-size: 12px; font-weight: 700; }
    input, textarea, select {
      width: 100%;
      margin-top: 4px;
      padding: 8px 9px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      font: inherit;
      font-size: 14px;
    }
    textarea { min-height: 78px; resize: vertical; }
    .toolbar { display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0 0; }
    .chat-topbar .toolbar, header .toolbar { margin: 0; }
    button {
      min-height: 34px;
      padding: 7px 11px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      font-weight: 700;
      cursor: pointer;
    }
    button.primary { border-color: var(--blue); background: var(--blue); color: #fff; }
    button.commit { border-color: var(--green); background: var(--green); color: #fff; }
    button.tool { border-color: var(--orange); color: var(--orange); }
    .messages {
      display: flex;
      flex-direction: column;
      gap: 18px;
      min-height: min-content;
      max-height: none;
      flex: 0 0 auto;
      overflow: visible;
      padding: 14px 64px 24px;
      border: 0;
      border-radius: 0;
      background: transparent;
    }
    .msg {
      position: relative;
      max-width: min(78%, 720px);
      padding: 12px 14px;
      border-radius: 16px;
      border: 0;
      white-space: pre-wrap;
      font-size: 15px;
      line-height: 1.65;
    }
    .msg.user {
      align-self: flex-end;
      border-bottom-right-radius: 6px;
      background: #eff2ff;
      color: #19223a;
    }
    .msg.assistant {
      align-self: flex-start;
      border-bottom-left-radius: 6px;
      background: #fff;
      box-shadow: 0 8px 28px rgba(43, 50, 77, 0.08);
    }
    .msg.tool {
      align-self: flex-start;
      background: #fff8ec;
      color: #8a4b12;
    }
    .msg .meta {
      margin-bottom: 5px;
      color: #8c95a7;
      font-size: 11px;
      font-weight: 850;
      text-transform: uppercase;
    }
    .composer-wrap {
      position: sticky;
      bottom: 0;
      padding: 0 48px 22px;
      background: linear-gradient(180deg, rgba(255,255,255,0), #fff 24%);
      z-index: 2;
    }
    .composer {
      border: 1px solid #e4e7ef;
      border-radius: 22px;
      background: #fff;
      box-shadow: 0 12px 36px rgba(28, 38, 68, 0.12);
      overflow: hidden;
    }
    .composer textarea {
      min-height: 78px;
      margin: 0;
      padding: 16px 18px 8px;
      border: 0;
      border-radius: 0;
      resize: none;
      font-size: 15px;
      line-height: 1.55;
      outline: none;
    }
    .composer-actions {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 8px 10px 10px 12px;
    }
    .tool-row {
      display: flex;
      flex-wrap: wrap;
      gap: 7px;
      min-width: 0;
    }
    .chip-button {
      min-height: 30px;
      padding: 5px 9px;
      border-color: #e7eaf2;
      border-radius: 999px;
      background: #f8f9fc;
      color: #596377;
      font-size: 12px;
      font-weight: 800;
    }
    .memory-test-select {
      width: 168px;
      min-height: 30px;
      margin: 0;
      padding: 5px 8px;
      border-radius: 999px;
      background: #fff;
      color: #465168;
      font-size: 12px;
      font-weight: 800;
    }
    .commit-memory-detail {
      display: grid;
      gap: 8px;
      margin-top: 12px;
      padding: 12px;
      border: 1px solid #dbe5f2;
      border-radius: 10px;
      background: #fbfcff;
      font-size: 13px;
      line-height: 1.5;
    }
    .commit-memory-empty { color: var(--muted); }
    .commit-memory-tags {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }
    .commit-memory-tag {
      padding: 4px 8px;
      border-radius: 999px;
      background: #edf4ff;
      color: #26508d;
      font-size: 12px;
      font-weight: 850;
    }
    .commit-memory-item {
      padding: 8px;
      border: 1px solid #e4e9f2;
      border-radius: 8px;
      background: #fff;
    }
    .commit-memory-item strong {
      display: inline;
      margin: 0;
      color: #22304a;
    }
    .send-button {
      display: grid;
      place-items: center;
      width: 36px;
      height: 36px;
      min-height: 36px;
      padding: 0;
      border: 0;
      border-radius: 50%;
      background: #111827;
      color: #fff;
      font-size: 18px;
      line-height: 1;
    }
    .manual-reply {
      display: none;
      padding: 0 48px 20px;
    }
    .manual-reply.open { display: block; }
    .manual-reply textarea { min-height: 70px; border-radius: 12px; }
    .inspect-grid {
      display: grid;
      grid-template-columns: 1fr;
      gap: 10px;
      height: calc(100vh - 140px);
      min-height: 606px;
    }
    .side-view { display: none; }
    #contextPanel.active {
      display: grid;
      grid-template-columns: 1fr;
      grid-template-rows: auto minmax(0, 1fr);
      gap: 10px;
      min-height: 0;
    }
    #memoryPanel.active {
      display: grid;
      grid-template-columns: 1fr;
      grid-auto-rows: max-content;
      align-content: start;
      gap: 10px;
      min-height: 0;
      overflow: auto;
    }
    .context-view {
      min-height: 0;
      max-height: none;
      overflow: auto;
      padding: 10px;
      border: 1px solid #e2e7f0;
      border-radius: 10px;
      background: #fbfcff;
      color: #243044;
      font-family: "Cascadia Code", Consolas, monospace;
      font-size: 12px;
      line-height: 1.55;
    }
    .context-message {
      margin-bottom: 10px;
      padding: 10px;
      border: 1px solid #e7ebf3;
      border-radius: 9px;
      background: #fff;
    }
    .context-role {
      margin-bottom: 6px;
      color: #51607a;
      font-family: "Segoe UI", "Microsoft YaHei", Arial, sans-serif;
      font-size: 12px;
      font-weight: 850;
    }
    .context-layer {
      display: inline-block;
      margin-left: 6px;
      padding: 2px 6px;
      border-radius: 999px;
      background: #eef2ff;
      color: #4050c9;
      font-size: 11px;
      font-weight: 900;
    }
    .context-layer.memory {
      background: #eaf3ff;
      color: #2458c7;
    }
    .context-content { white-space: pre-wrap; }
    .memory-highlight {
      display: block;
      margin: 6px 0;
      padding: 10px;
      border: 1px solid #bdd7ff;
      border-left: 4px solid #4f7cff;
      border-radius: 8px;
      background: #f0f6ff;
      color: #173b75;
      font-weight: 800;
    }
    .memory-highlight::before {
      content: "EchoMemory 检索上下文";
      display: block;
      margin-bottom: 6px;
      color: #315fbd;
      font-family: "Segoe UI", "Microsoft YaHei", Arial, sans-serif;
      font-size: 11px;
      font-weight: 900;
      text-transform: uppercase;
    }
    .history-highlight {
      display: block;
      margin: 6px 0;
      padding: 10px;
      border: 1px solid #cde8d7;
      border-left: 4px solid #35a76c;
      border-radius: 8px;
      background: #f0fbf4;
      color: #145c3a;
      font-weight: 800;
    }
    .fs-controls {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 84px;
      gap: 8px;
      align-items: end;
    }
    .fs-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 8px;
    }
    .fs-view {
      min-height: 170px;
      max-height: 310px;
      overflow: auto;
      padding: 8px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      font-family: "Cascadia Code", Consolas, monospace;
      font-size: 12px;
    }
    .fs-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 10px;
      align-items: center;
      min-height: 28px;
      padding: 4px 6px;
      border-radius: 5px;
      cursor: default;
    }
    .fs-row:hover { background: #eef4ff; }
    .fs-row.file { cursor: pointer; }
    .fs-name {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .fs-meta { color: var(--muted); font-size: 11px; }
    .fs-empty { color: var(--muted); padding: 8px; }
    pre {
      min-height: 118px;
      max-height: 280px;
      overflow: auto;
      margin: 0;
      padding: 10px;
      border-radius: 6px;
      border: 1px solid #e2e7f0;
      background: #fbfcff;
      color: #263245;
      font-family: "Cascadia Code", Consolas, monospace;
      font-size: 12px;
      line-height: 1.45;
    }
    .error { color: var(--red); font-weight: 700; }
    @media (max-width: 960px) {
      main { grid-template-columns: 1fr; }
      .row { grid-template-columns: 1fr; }
      .chat-shell { grid-template-columns: 1fr; height: auto; min-height: 620px; }
      .chat-nav { display: none; }
      .messages { padding: 12px 18px 22px; }
      .composer-wrap, .manual-reply { padding-left: 16px; padding-right: 16px; }
      .identity-row { grid-template-columns: 1fr; }
      .memory-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>EchoMemory 智能体对话</h1>
      <div class="status" id="status">连接中...</div>
    </div>
    <div class="toolbar">
      <button id="refresh">刷新状态</button>
    </div>
  </header>
  <main>
    <section class="chat-section">
      <div class="chat-shell">
        <aside class="chat-nav">
          <div class="brand-row">
            <div class="brand-dot">E</div>
            <div>Echo 智能体</div>
          </div>
          <button class="nav-button primary" id="openSession">＋ 新对话</button>
          <div class="nav-list">
            <button class="nav-item active" id="chatNav">◇ 对话</button>
            <button class="nav-item" id="memoryNav">◎ EchoMemory</button>
          </div>
          <div class="nav-section-title">最近会话</div>
          <div id="sessionList"></div>
          <div class="chat-nav-footer">
            Alibaba / Qwen 已接入<br />
            EchoMemory 写入与检索实时生效
          </div>
        </aside>
        <div class="chat-main" id="chatMain">
          <div class="main-view" id="chatMainView">
          <div>
            <div class="chat-topbar">
              <div class="chat-title">
                <span>EchoMemory Agent</span>
                <span class="model-pill" id="modelBadge">模型加载中</span>
              </div>
            </div>
            <div class="identity-row">
              <label>user_id<input id="userId" value="alice" /></label>
              <label>agent_id<input id="agentId" value="demo-agent" /></label>
              <input id="sessionId" type="hidden" />
            </div>
          </div>
          <div class="chat-scroll" id="chatScroll">
            <div class="welcome" id="welcome">
              <div>
                <h2>今天想聊点什么？</h2>
                <p>发起一轮真实 Agent 对话，消息会写入 EchoMemory 并调用 Alibaba 模型。</p>
                <div class="quick-prompts">
                  <button class="quickPrompt">帮我整理 D03 的提交方案</button>
                  <button class="quickPrompt">总结当前会话中的关键记忆</button>
                  <button class="quickPrompt">生成一个三步执行计划</button>
                </div>
              </div>
            </div>
            <div class="messages" id="messages"></div>
          </div>
          <div>
            <div class="composer-wrap">
              <div class="composer">
                <textarea id="userText" placeholder="给 EchoMemory Agent 发送消息">帮我整理一下 D03 的提交方案</textarea>
                <div class="composer-actions">
                  <div class="tool-row">
                    <button class="chip-button" id="memoryChip">记忆检索</button>
                    <select class="memory-test-select" id="memoryTestKind" title="选择记忆测试场景">
                      <option value="all">全部类型记忆</option>
                      <option value="profile">profile 用户画像</option>
                      <option value="preference">preference 偏好</option>
                      <option value="entity">entity 实体</option>
                      <option value="event">event 事件</option>
                      <option value="agent_case">agent_case 案例</option>
                      <option value="pattern">pattern 模式</option>
                      <option value="tool_experience">tool_experience 工具经验</option>
                    </select>
                    <button class="chip-button" id="memoryTestChip">记忆测试</button>
                    <button class="chip-button" id="commitChip">提交归档</button>
                  </div>
                  <button class="send-button" id="sendChat" title="发送并生成">↑</button>
                </div>
              </div>
            </div>
          </div>
          </div>
          <div class="memory-main" id="memoryMainView">
            <div class="chat-topbar">
              <div class="chat-title">
                <span>EchoMemory</span>
                <span class="model-pill">记忆运行时</span>
              </div>
            </div>
            <div class="memory-page">
              <div class="memory-hero">
                <h2>EchoMemory 状态</h2>
                <p>这里聚合当前 Agent 使用的记忆配置、session 文件树、事件和最后一次接口响应。对话窗口只在「对话」页显示。</p>
              </div>
              <div class="memory-grid">
                <div class="memory-card">
                  <strong>当前会话</strong>
                  <span id="memorySessionLabel">-</span>
                </div>
                <div class="memory-card">
                  <strong>模型</strong>
                  <span id="memoryModelLabel">-</span>
                </div>
                <div class="memory-card">
                  <strong>EchoMemory Target</strong>
                  <span id="memoryTargetLabel">echo://sessions/-</span>
                </div>
                <div class="memory-card">
                  <strong>操作</strong>
                  <button class="chip-button" id="memoryRefresh">刷新 EchoMemory 信息</button>
                </div>
              </div>
              <div class="commit-memory-detail" id="commitMemorySummary">
                <div class="commit-memory-empty">完成提交后，这里会显示本次 commit 实际抽取的记忆类型和详情。</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
    <section>
      <div class="panel-head">
        <h2 id="sideTitle">上下文检查器</h2>
        <span class="status" id="sideStatus">模型上下文</span>
      </div>
      <div class="panel-body inspect-grid">
        <div class="side-view active" id="contextPanel">
          <label>组装后的模型上下文</label>
          <div id="contextView" class="context-view">发送消息后，这里会显示本轮实际拼接并发送给模型的上下文。</div>
        </div>
        <div class="side-view" id="memoryPanel">
          <label>智能体配置</label>
          <pre id="config">{}</pre>
          <label>EchoMemory Runtime</label>
          <pre id="runtime">{}</pre>
          <label>文件系统目标</label>
          <div class="fs-controls">
            <label>目标 URI<input id="fsTarget" value="echo://sessions/chat-001" /></label>
            <label>深度<input id="fsDepth" value="5" /></label>
          </div>
          <div class="fs-actions">
            <button id="treeMode">树状展开</button>
            <button id="flatMode">平铺展开</button>
            <button id="refreshTree">刷新目标</button>
            <button id="treeEngineTarget">Tree 引擎</button>
          </div>
          <div class="fs-view" id="tree"></div>
          <label>Selected File</label>
          <pre id="fileContent">点击文件查看内容</pre>
          <label>Events</label>
          <pre id="events">{}</pre>
          <label>Commit Memory Details</label>
          <div class="commit-memory-detail" id="commitMemoryDetails">
            <div class="commit-memory-empty">暂无 commit 记忆详情。</div>
          </div>
          <label>Last Response</label>
          <pre id="last">{}</pre>
        </div>
      </div>
    </section>
  </main>
  <script>
    const $ = (id) => document.getElementById(id);
    let currentTree = {uri: "", entries: []};
    let fsMode = "tree";
    let sessions = [];
    let activeSession = null;
    let lastCommitMemorySummary = null;
    const memoryTestScenarios = {
      profile: ["我是后端工程师，长期维护 EchoMemory 记忆系统。"],
      preference: ["我喜欢你以后默认先给结论，再给简洁步骤。"],
      entity: ["EchoMemory 项目中的 TreeMemoryEngine 模块负责把会话归档抽取成结构化记忆。"],
      event: ["决定先完成 simple 引擎验证，再推进召回优化。"],
      agent_case: ["这个任务的问题是 commit 结果看不到记忆类型，修复后结果是页面可以展示本轮抽取详情。"],
      pattern: ["每次修改公开接口后的固定流程是补单测、跑完整测试、再重启服务。"],
      tool_experience: ["使用 rg 命令定位文本，遇到路径过宽时先收窄目录再重试。"],
      all: [
        "我是后端工程师，目标是把 EchoMemory 做成稳定的记忆系统。",
        "我喜欢你以后默认用简洁中文和分步骤说明。",
        "TreeMemoryEngine 模块负责把会话归档抽取成结构化记忆。",
        "决定先完成 simple 引擎验证，再推进召回优化。",
        "这个任务的问题是接口返回不透明，修复后结果是可以看到 commit 记忆类型。",
        "每次修改公开接口后的固定流程是补单测、跑测试、再重启服务。",
        "使用 rg 命令定位文本，遇到参数报错时先收窄路径再重试。"
      ]
    };
    const sessionStoreKey = "echomem-agent.sessions.v1";

    function sessionId() { return $("sessionId").value.trim(); }
    function createSessionId() {
      const stamp = new Date().toISOString().replace(/[-:.TZ]/g, "").slice(0, 14);
      const suffix = Math.random().toString(36).slice(2, 8);
      return `chat-${stamp}-${suffix}`;
    }
    function ensureSessionId() {
      if (!activeSession) createConversation();
      if (!sessionId()) $("sessionId").value = activeSession.id;
      syncFsTarget(true);
      return sessionId();
    }
    function show(id, value) { $(id).textContent = JSON.stringify(value, null, 2); }
    function findSession(id) {
      return sessions.find((item) => item.id === id) || null;
    }
    function loadSessions() {
      try {
        const parsed = JSON.parse(localStorage.getItem(sessionStoreKey) || "[]");
        sessions = Array.isArray(parsed) ? parsed : [];
      } catch {
        sessions = [];
      }
    }
    function saveSessions() {
      localStorage.setItem(sessionStoreKey, JSON.stringify(sessions.slice(0, 30)));
    }
    function createConversation() {
      activeSession = {
        id: createSessionId(),
        title: "新对话",
        messages: [],
        context: [],
        contextTrace: null,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
      };
      sessions = [activeSession, ...sessions.filter((item) => item.id !== activeSession.id)];
      $("sessionId").value = activeSession.id;
      saveSessions();
      renderSessions();
      return activeSession;
    }
    function setActiveSession(session) {
      activeSession = session;
      $("sessionId").value = session.id;
      renderSessions();
      renderConversation();
      syncFsTarget(true);
      refreshTree();
    }
    function deleteSession(id) {
      const next = sessions.filter((item) => item.id !== id);
      sessions = next;
      if (activeSession?.id === id) {
        activeSession = sessions[0] || null;
        if (!activeSession) createConversation();
        else $("sessionId").value = activeSession.id;
        renderConversation();
        syncFsTarget(true);
      }
      saveSessions();
      renderSessions();
    }
    function renderSessions() {
      const list = $("sessionList");
      list.innerHTML = "";
      if (sessions.length === 0) {
        const empty = document.createElement("div");
        empty.className = "recent-chat";
        empty.innerHTML = `<span class="recent-title">暂无会话</span>`;
        list.appendChild(empty);
        return;
      }
      for (const session of sessions) {
        const row = document.createElement("div");
        row.className = `recent-chat ${activeSession?.id === session.id ? "active" : ""}`;
        row.innerHTML = `
          <span class="recent-title">${escapeHtml(session.title || session.id)}</span>
          <button class="delete-session" title="删除会话">×</button>
        `;
        row.onclick = () => setActiveSession(session);
        row.querySelector(".delete-session").onclick = (event) => {
          event.stopPropagation();
          deleteSession(session.id);
        };
        list.appendChild(row);
      }
    }
    function renderConversation() {
      $("messages").innerHTML = "";
      const items = activeSession?.messages || [];
      $("welcome").style.display = items.length === 0 ? "grid" : "none";
      for (const item of items) bubble(item.role, item.content, {persist: false});
      if (activeSession?.context?.length) showContext(activeSession.context, activeSession.contextTrace);
      else $("contextView").textContent = "发送消息后，这里会显示本轮实际拼接并发送给模型的上下文。";
    }
    function rememberMessage(role, content) {
      rememberMessageFor(sessionId(), role, content);
    }
    function rememberMessageFor(targetSessionId, role, content) {
      const session = findSession(targetSessionId);
      if (!session) return;
      session.messages.push({role, content});
      session.updatedAt = new Date().toISOString();
      sessions = [session, ...sessions.filter((item) => item.id !== session.id)];
      if (activeSession?.id === session.id) activeSession = session;
      saveSessions();
      renderSessions();
    }
    function rememberContext(messages, trace = null) {
      rememberContextFor(sessionId(), messages, trace);
    }
    function rememberContextFor(targetSessionId, messages, trace = null) {
      const session = findSession(targetSessionId);
      if (!session) return;
      session.context = messages;
      session.contextTrace = trace;
      session.updatedAt = new Date().toISOString();
      sessions = [session, ...sessions.filter((item) => item.id !== session.id)];
      if (activeSession?.id === session.id) activeSession = session;
      saveSessions();
      renderSessions();
    }
    function showContext(messages, trace = null) {
      if (!Array.isArray(messages)) return;
      const rows = contextDisplayRows(messages, trace);
      $("contextView").innerHTML = rows.map((row) => `
        <div class="context-message">
          <div class="context-role">${escapeHtml(row.title)}${contextLayerBadge(row.layer)}</div>
          <div class="context-content">${highlightContext(row.content || "", row.role)}</div>
        </div>
      `).join("");
    }
    function contextDisplayRows(messages, trace) {
      const layers = Array.isArray(trace?.layers) ? trace.layers : [];
      const grouped = new Set();
      const rows = [];
      for (const layer of layers) {
        if (layer.name !== "近期对话" || !Array.isArray(layer.message_indexes) || layer.message_indexes.length <= 1) continue;
        for (const index of layer.message_indexes) grouped.add(index);
        rows.push({
          title: `#${layer.message_indexes[0] + 1}-${layer.message_indexes[layer.message_indexes.length - 1] + 1} 近期对话`,
          role: "conversation",
          layer,
          content: layer.message_indexes.map((index) => {
            const message = messages[index] || {};
            const speaker = message.role === "assistant" ? "助手" : "用户";
            return `【${speaker}】\n${message.content || ""}`;
          }).join("\n\n")
        });
      }
      for (let index = 0; index < messages.length; index += 1) {
        if (grouped.has(index)) continue;
        const message = messages[index] || {};
        rows.push({
          title: `#${index + 1} ${roleLabel(message.role)}`,
          role: message.role || "",
          layer: contextLayerFor(index, trace),
          content: message.content || ""
        });
      }
      return rows.sort((a, b) => firstMessageIndex(a.layer, a.title) - firstMessageIndex(b.layer, b.title));
    }
    function firstMessageIndex(layer, title) {
      if (Array.isArray(layer?.message_indexes) && layer.message_indexes.length > 0) return layer.message_indexes[0];
      const match = String(title || "").match(/^#(\d+)/);
      return match ? Number(match[1]) - 1 : 0;
    }
    function roleLabel(role) {
      if (role === "system") return "系统";
      if (role === "user") return "用户";
      if (role === "assistant") return "助手";
      if (role === "tool") return "工具";
      return role || "消息";
    }
    function contextLayerFor(index, trace) {
      const layers = Array.isArray(trace?.layers) ? trace.layers : [];
      return layers.find((item) => Array.isArray(item.message_indexes) && item.message_indexes.includes(index));
    }
    function contextLayerBadge(layer) {
      if (!layer) return "";
      const isMemory = layer.source === "EchoMemory";
      const label = `${layer.name}${layer.source ? " / " + layer.source : ""}`;
      return `<span class="context-layer ${isMemory ? "memory" : ""}">${escapeHtml(label)}</span>`;
    }
    function highlightContext(content, role) {
      const escaped = escapeHtml(content);
      const withMemory = escaped.replace(/(&lt;retrieved_memory\b[\s\S]*?&lt;\/retrieved_memory&gt;|&lt;memory&gt;[\s\S]*?&lt;\/memory&gt;)/g, '<strong class="memory-highlight">$1</strong>');
      if (role === "user" || /(&lt;history&gt;|&lt;session&gt;|&lt;current_request&gt;)/.test(withMemory)) {
        return `<strong class="history-highlight">${withMemory}</strong>`;
      }
      return withMemory;
    }
    function resetConversation() {
      createConversation();
      $("messages").innerHTML = "";
      $("welcome").style.display = "grid";
      $("contextView").textContent = "新对话已创建。发送消息后，这里会显示本轮实际拼接并发送给模型的上下文。";
      syncFsTarget(true);
    }
    function bubble(role, text, options = {}) {
      $("welcome").style.display = "none";
      const div = document.createElement("div");
      div.className = `msg ${role}`;
      div.innerHTML = `<div class="meta">${roleLabel(role)}</div>${escapeHtml(text)}`;
      $("messages").appendChild(div);
      $("chatScroll").scrollTop = $("chatScroll").scrollHeight;
      if (options.persist !== false) rememberMessage(role, text);
    }
    function appendBubbleToCurrent(role, text, options = {}) {
      if (!activeSession) createConversation();
      bubble(role, text, options);
    }
    function escapeHtml(text) {
      return text.replace(/[&<>"']/g, (ch) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
      }[ch]));
    }
    function syncFsTarget(force = false) {
      const expected = "echo://sessions/" + sessionId();
      if (force || !$("fsTarget").value.trim() || $("fsTarget").value.includes("chat-001")) {
        $("fsTarget").value = expected;
      }
    }
    function setSideView(view) {
      const isMemory = view === "memory";
      $("contextPanel").classList.toggle("active", !isMemory);
      $("memoryPanel").classList.toggle("active", isMemory);
      $("chatMainView").classList.toggle("hidden", isMemory);
      $("memoryMainView").classList.toggle("active", isMemory);
      $("chatNav").classList.toggle("active", !isMemory);
      $("memoryNav").classList.toggle("active", isMemory);
      $("sideTitle").textContent = isMemory ? "EchoMemory" : "上下文检查器";
      $("sideStatus").textContent = isMemory ? "记忆与调试信息" : "模型上下文";
      if (isMemory) refreshInspectors();
    }
    function depthFor(entry) {
      const rootParts = currentTree.uri.replace("echo://", "").split("/").filter(Boolean);
      const parts = entry.uri.replace("echo://", "").split("/").filter(Boolean);
      return Math.max(0, parts.length - rootParts.length);
    }
    function renderFs(entries) {
      const container = $("tree");
      container.innerHTML = "";
      if (!entries || entries.length === 0) {
        container.innerHTML = `<div class="fs-empty">目标下暂无内容</div>`;
        return;
      }
      const rows = fsMode === "flat" ? entries : entries.slice().sort((a, b) => a.uri.localeCompare(b.uri));
      for (const entry of rows) {
        const row = document.createElement("div");
        row.className = `fs-row ${entry.kind === "file" ? "file" : "directory"}`;
        const indent = fsMode === "tree" ? depthFor(entry) * 16 : 0;
        const mark = entry.kind === "directory" ? "[D]" : "[F]";
        row.innerHTML = `
          <div class="fs-name" style="padding-left:${indent}px">${mark} ${escapeHtml(fsMode === "flat" ? entry.uri : entry.name)}</div>
          <div class="fs-meta">${entry.kind}${entry.kind === "file" ? " · " + entry.size + "B" : ""}</div>
        `;
        if (entry.kind === "file") {
          row.onclick = () => readFile(entry.uri);
          row.title = "点击读取文件";
        }
        container.appendChild(row);
      }
    }
    async function refreshTree() {
      try {
        const uri = $("fsTarget").value.trim();
        const depth = $("fsDepth").value.trim() || "5";
        currentTree = await fetch(`/agent/inspect/fs/tree?uri=${encodeURIComponent(uri)}&max_depth=${encodeURIComponent(depth)}`).then((r) => r.json());
        if (currentTree.error) throw new Error(currentTree.message || currentTree.error);
        renderFs(currentTree.entries);
      } catch (error) {
        $("tree").innerHTML = `<div class="fs-empty error">${escapeHtml(error.message)}</div>`;
      }
    }
    async function readFile(uri) {
      try {
        const data = await fetch(`/agent/inspect/fs/read?uri=${encodeURIComponent(uri)}`).then((r) => r.json());
        if (data.error) throw new Error(data.message || data.error);
        $("fileContent").textContent = data.text;
        show("last", data);
      } catch (error) {
        $("fileContent").textContent = error.message;
      }
    }
    async function request(path, options = {}) {
      const response = await fetch(path, {
        headers: {"Content-Type": "application/json"},
        ...options
      });
      const data = await response.json();
      show("last", data);
      if (!response.ok) throw new Error(data.message || data.error || response.statusText);
      await refreshInspectors();
      return data;
    }
    async function waitCommit(sessionIdForCommit, archiveId) {
      for (let attempt = 0; attempt < 240; attempt += 1) {
        const response = await fetch(`/api/sessions/${encodeURIComponent(sessionIdForCommit)}/commits/${encodeURIComponent(archiveId)}`);
        const data = await response.json();
        show("last", data);
        if (!response.ok || data.error) throw new Error(data.message || data.error || `commit status ${response.status}`);
        if (data.status?.status === "completed") return data.status;
        if (data.status?.status === "failed") throw new Error(data.status.error || "commit failed");
        if (attempt % 10 === 0) $("sideStatus").textContent = `等待 commit 完成：${archiveId}`;
        await new Promise((resolve) => setTimeout(resolve, 500));
      }
      throw new Error(`commit timeout: ${archiveId}`);
    }
    async function fetchCommitMemories(sessionIdForCommit, commitId) {
      const data = await fetch(`/api/sessions/${encodeURIComponent(sessionIdForCommit)}/commits/${encodeURIComponent(commitId)}/memories`).then((r) => r.json());
      show("last", data);
      if (data.error) throw new Error(data.message || data.error);
      return data.summary || {memory_kinds: [], memories: []};
    }
    function formatCommitMemories(summary) {
      const kinds = Array.isArray(summary.memory_kinds) ? summary.memory_kinds : [];
      if (kinds.length === 0) return "本次 commit 没有抽取出长期记忆。";
      const count = Array.isArray(summary.memories) ? summary.memories.length : 0;
      return `本次 commit 抽取了 ${count} 条记忆，类型：${kinds.join(", ")}`;
    }
    function renderCommitMemoryDetails(summary) {
      lastCommitMemorySummary = summary;
      const html = commitMemoryHtml(summary);
      $("commitMemorySummary").innerHTML = html;
      $("commitMemoryDetails").innerHTML = html;
    }
    function commitMemoryHtml(summary) {
      if (!summary) return `<div class="commit-memory-empty">暂无 commit 记忆详情。</div>`;
      const kinds = Array.isArray(summary.memory_kinds) ? summary.memory_kinds : [];
      const memories = Array.isArray(summary.memories) ? summary.memories : [];
      if (kinds.length === 0) return `<div class="commit-memory-empty">本次 commit 没有抽取出长期记忆。</div>`;
      const tags = kinds.map((kind) => `<span class="commit-memory-tag">${escapeHtml(kind)}</span>`).join("");
      const items = memories.map((memory) => `
        <div class="commit-memory-item">
          <strong>${escapeHtml(memory.kind || "memory")}</strong>
          ${escapeHtml(memory.title || "")}
          <div>${escapeHtml(memory.text || "")}</div>
        </div>
      `).join("");
      return `
        <div><strong>commit:</strong> ${escapeHtml(summary.commit_id || "-")}</div>
        <div class="commit-memory-tags">${tags}</div>
        ${items}
      `;
    }
    async function refreshInspectors() {
      try {
        const config = await fetch("/agent/config").then((r) => r.json());
        show("config", config);
        $("modelBadge").textContent = `${config.model?.provider || "模型"} / ${config.model?.model || "未知"}`;
        $("memoryModelLabel").textContent = `${config.model?.provider || "模型"} / ${config.model?.model || "未知"}`;
        $("status").textContent = `模型 ${config.model?.provider || ""} / EchoMemory ${sessionId()}`;
      } catch (error) {
        show("config", {error: error.message});
        $("modelBadge").textContent = "模型不可用";
        $("status").innerHTML = `<span class="error">${escapeHtml(error.message)}</span>`;
      }
      try {
        const runtime = await fetch("/agent/inspect/runtime").then((r) => r.json());
        show("runtime", runtime);
      } catch (error) {
        show("runtime", {error: error.message});
      }
      syncFsTarget();
      $("memorySessionLabel").textContent = sessionId() || "-";
      $("memoryTargetLabel").textContent = $("fsTarget").value || "echo://sessions/-";
      await refreshTree();
      try {
        show("events", await fetch("/agent/inspect/events").then((r) => r.json()));
      } catch (error) {
        show("events", {error: error.message});
      }
    }
    $("openSession").onclick = async () => {
      resetConversation();
    };
    async function sendChatMessage(content, options = {}) {
      const activeSessionId = ensureSessionId();
      const targetSession = findSession(activeSessionId);
      if (targetSession && targetSession.title === "新对话") {
        targetSession.title = content;
        if (activeSession?.id === targetSession.id) activeSession = targetSession;
        saveSessions();
        renderSessions();
      }
      bubble("user", content, {persist: options.persistUser !== false});
      const pending = document.createElement("div");
      pending.className = "msg assistant";
      pending.innerHTML = `<div class="meta">助手</div>生成中...`;
      $("messages").appendChild(pending);
      $("chatScroll").scrollTop = $("chatScroll").scrollHeight;
      try {
        const data = await request("/agent/chat", {
          method: "POST",
          body: JSON.stringify({
            user_id: $("userId").value.trim(),
            agent_id: $("agentId").value.trim(),
            session_id: activeSessionId,
            message: content,
            stream: false
          })
        });
        rememberContextFor(activeSessionId, data.messages, data.context_trace);
        rememberMessageFor(activeSessionId, "assistant", data.assistant.content);
        if (activeSession?.id === activeSessionId) {
          if (pending.isConnected) {
            pending.innerHTML = `<div class="meta">助手</div>${escapeHtml(data.assistant.content)}`;
            showContext(data.messages, data.context_trace);
          } else {
            renderConversation();
          }
        }
        return data;
      } catch (error) {
        rememberMessageFor(activeSessionId, "tool", `请求失败：${error.message}`);
        if (activeSession?.id === activeSessionId) {
          if (pending.isConnected) {
            pending.className = "msg tool";
            pending.innerHTML = `<div class="meta">error</div>${escapeHtml(error.message)}`;
          } else {
            renderConversation();
          }
        }
        throw error;
      }
    }
    $("sendChat").onclick = async () => {
      const content = $("userText").value.trim();
      if (!content) return;
      $("userText").value = "";
      await sendChatMessage(content);
    };
    async function commitCurrentSession() {
      const sessionIdForCommit = ensureSessionId();
      const data = await request(`/api/sessions/${encodeURIComponent(sessionIdForCommit)}/commit`, {
        method: "POST",
        body: "{}"
      });
      if (activeSession?.id === sessionIdForCommit) bubble("assistant", `commit accepted -> ${data.result.archive_id}`);
      else rememberMessageFor(sessionIdForCommit, "assistant", `commit accepted -> ${data.result.archive_id}`);
      const commitId = data.result.commit_id || data.result.archive_id;
      const status = await waitCommit(sessionIdForCommit, commitId);
      const memorySummary = await fetchCommitMemories(sessionIdForCommit, commitId);
      renderCommitMemoryDetails(memorySummary);
      await refreshInspectors();
      const message = `commit ${status.status} -> ${commitId}\n${formatCommitMemories(memorySummary)}`;
      if (activeSession?.id === sessionIdForCommit) bubble("assistant", message);
      else rememberMessageFor(sessionIdForCommit, "assistant", message);
      return memorySummary;
    }
    async function runMemoryTest() {
      const kind = $("memoryTestKind").value || "all";
      const scenario = memoryTestScenarios[kind] || memoryTestScenarios.all;
      $("memoryTestChip").disabled = true;
      $("commitChip").disabled = true;
      try {
        setSideView("context");
        bubble("tool", `开始记忆测试：${kind}，共 ${scenario.length} 轮。`);
        for (const content of scenario) {
          await sendChatMessage(content);
        }
        bubble("tool", "测试对话完成，开始自动提交并抽取记忆。");
        const summary = await commitCurrentSession();
        setSideView("memory");
        show("last", {memory_test: kind, summary});
      } catch (error) {
        bubble("tool", `记忆测试失败：${error.message}`);
      } finally {
        $("memoryTestChip").disabled = false;
        $("commitChip").disabled = false;
      }
    }
    $("refresh").onclick = refreshInspectors;
    $("refreshTree").onclick = refreshTree;
    $("treeMode").onclick = () => { fsMode = "tree"; renderFs(currentTree.entries); };
    $("flatMode").onclick = () => { fsMode = "flat"; renderFs(currentTree.entries); };
    $("treeEngineTarget").onclick = async () => {
      $("fsTarget").value = "echo://engine/tree";
      await refreshTree();
    };
    $("chatNav").onclick = () => setSideView("context");
    $("memoryNav").onclick = () => setSideView("memory");
    $("memoryRefresh").onclick = refreshInspectors;
    $("memoryChip").onclick = () => $("userText").value = "请基于 EchoMemory 检索结果，帮我总结当前会话中的关键记忆。";
    $("memoryTestChip").onclick = runMemoryTest;
    $("commitChip").onclick = commitCurrentSession;
    $("sessionId").oninput = syncFsTarget;
    for (const item of document.querySelectorAll(".quickPrompt")) {
      item.onclick = () => {
        $("userText").value = item.textContent;
        $("userText").focus();
      };
    }
    $("userText").addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        $("sendChat").click();
      }
    });
    loadSessions();
    if (sessions.length > 0) setActiveSession(sessions[0]);
    else resetConversation();
    refreshInspectors();
  </script>
</body>
</html>"""


class AgentPlaygroundServer(ThreadingHTTPServer):
    """HTTP server for the standalone agent playground."""

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        *,
        config: AgentConfig,
    ) -> None:
        super().__init__(server_address, handler_class)
        self.config = config
        self.echomem_url = config.echomemory.base_url.rstrip("/") + "/"


class AgentRequestHandler(BaseHTTPRequestHandler):
    """Serve the playground page and proxy EchoMemory HTTP calls."""

    server_version = "EchoMemAgent/0.1"

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._send_html(INDEX_HTML)
            return
        if path == "/agent/config":
            self._send_json(HTTPStatus.OK, self.server.config.public_dict())
            return
        if path.startswith("/agent/inspect/") or path.startswith("/api/"):
            self._proxy()
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/agent/chat":
            self._agent_chat()
            return
        if path == "/agent/session/commit":
            self._agent_commit()
            return
        if path.startswith("/api/"):
            self._proxy()
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def log_message(self, format: str, *args: Any) -> None:
        """Silence default access logs during normal operation."""

    def _proxy(self) -> None:
        try:
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            target = urljoin(self.server.echomem_url, self.path.lstrip("/"))
            request = Request(
                target,
                data=body if self.command in {"POST", "PUT", "PATCH"} else None,
                method=self.command,
                headers={"Content-Type": self.headers.get("Content-Type", "application/json")},
            )
            with urlopen(request, timeout=5) as response:
                response_body = response.read()
                self._send_proxy_response(response.status, response_body)
        except HTTPError as exc:
            self._send_proxy_response(exc.code, exc.read())
        except URLError as exc:
            self._send_json(
                HTTPStatus.BAD_GATEWAY,
                {"error": "echomem_unreachable", "message": str(exc.reason)},
            )

    def _agent_chat(self) -> None:
        try:
            payload = self._read_json()
            data = AgentChatService(self.server.config).chat(payload)
            self._send_json(HTTPStatus.OK, data)
        except ValueError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": str(exc)})
        except EchoMemoryClientError as exc:
            self._send_json(HTTPStatus.BAD_GATEWAY, {"error": "echomemory_error", "message": str(exc)})
        except ModelClientError as exc:
            self._send_json(HTTPStatus.BAD_GATEWAY, {"error": "model_error", "message": str(exc)})

    def _agent_commit(self) -> None:
        try:
            payload = self._read_json()
            session_id = str(payload.get("session_id") or "").strip()
            if not session_id:
                raise ValueError("session_id is required")
            data = AgentChatService(self.server.config).memory.commit(session_id)
            self._send_json(HTTPStatus.OK, data)
        except ValueError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": str(exc)})
        except EchoMemoryClientError as exc:
            self._send_json(HTTPStatus.BAD_GATEWAY, {"error": "echomemory_error", "message": str(exc)})

    def _read_json(self) -> dict[str, Any]:
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        if not body:
            return {}
        try:
            data = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("request body must be valid JSON") from exc
        if not isinstance(data, dict):
            raise ValueError("request body must be a JSON object")
        return data

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self._send_proxy_response(status.value, body)

    def _send_proxy_response(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def create_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    echomem_url: str | None = None,
    config_path: str | None = None,
) -> AgentPlaygroundServer:
    """Create the standalone agent playground server."""
    config = load_config(config_path, echomem_url=echomem_url)
    return AgentPlaygroundServer((host, port), AgentRequestHandler, config=config)


def serve(host: str = "127.0.0.1", port: int = 8765, echomem_url: str | None = None) -> None:
    """Run the agent playground until interrupted."""
    server = create_server(host, port, echomem_url=echomem_url)
    try:
        print(f"agent playground listening on http://{host}:{server.server_port}")
        print(f"proxying EchoMemory at {server.echomem_url.rstrip('/')}")
        server.serve_forever()
    except KeyboardInterrupt:
        print("agent playground stopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    serve()
