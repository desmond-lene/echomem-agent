"""Standalone agent playground server.

This package intentionally avoids importing echomem internals. It can be moved
to a separate project and continue to work against the EchoMemory HTTP API.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urljoin, urlparse
from urllib.request import Request, urlopen

from .chat_service import AgentChatService
from .config import AgentConfig, load_config
from .echomemory_client import EchoMemoryClientError
from .graph_eval_service import GraphEvalService
from .locomo_dataset import LocomoDatasetError, LocomoDatasetService
from .locomo_eval_service import LocomoEvalError, LocomoEvalService
from .model_client import ModelClientError
INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>EchoMemory 閺呴缚鍏樻担鎾愁嚠鐠?/title>
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
    .header-title {
      display: grid;
      gap: 3px;
      min-width: 220px;
    }
    .header-actions {
      display: flex;
      align-items: center;
      gap: 10px;
      min-width: 0;
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
    .account-switcher {
      display: grid;
      grid-template-columns: auto minmax(120px, 180px) minmax(180px, 240px) auto auto;
      gap: 10px;
      align-items: center;
      min-width: 0;
      padding: 6px 8px;
      border: 1px solid #e4e7ef;
      border-radius: 10px;
      background: #fafbfe;
    }
    .account-label {
      color: #7a8495;
      font-size: 12px;
      font-weight: 800;
    }
    .current-account {
      max-width: 180px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      color: #25314a;
      font-size: 13px;
      font-weight: 850;
    }
    .account-switcher select {
      width: 100%;
      min-height: 34px;
      border: 1px solid #e4e7ef;
      border-radius: 10px;
      background: #fafbfe;
      color: var(--ink);
      padding: 0 10px;
    }
    .account-switcher button[disabled] {
      cursor: not-allowed;
      opacity: 0.58;
    }
    .modal-backdrop {
      position: fixed;
      inset: 0;
      display: none;
      align-items: center;
      justify-content: center;
      padding: 20px;
      background: rgba(15, 23, 42, 0.38);
      z-index: 50;
    }
    .modal-backdrop.open { display: flex; }
    .modal-card {
      width: min(100%, 420px);
      padding: 18px;
      border: 1px solid #dce3f1;
      border-radius: 16px;
      background: #fff;
      box-shadow: 0 24px 60px rgba(15, 23, 42, 0.18);
    }
    .modal-card h3 {
      margin: 0 0 6px;
      font-size: 18px;
    }
    .modal-card p {
      margin: 0 0 12px;
      color: #6c7587;
      font-size: 13px;
    }
    .modal-actions {
      display: flex;
      justify-content: flex-end;
      gap: 8px;
      margin-top: 14px;
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
    .context-toggle {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      min-height: 30px;
      padding: 5px 9px;
      border: 1px solid #e7eaf2;
      border-radius: 999px;
      background: #fff;
      color: #465168;
      font-size: 12px;
      font-weight: 800;
      white-space: nowrap;
    }
    .context-toggle input {
      width: 14px;
      height: 14px;
      margin: 0;
      accent-color: var(--blue);
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
      content: "EchoMemory 濡偓缁鳖澀绗傛稉瀣瀮";
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
      header { align-items: stretch; flex-direction: column; }
      .header-actions { flex-wrap: wrap; }
      .account-switcher { grid-template-columns: 1fr; width: 100%; }
      .memory-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <div class="header-title">
      <h1>EchoMemory 閺呴缚鍏樻担鎾愁嚠鐠?/h1>
      <div class="status" id="status">鏉╃偞甯存稉?..</div>
    </div>
    <div class="header-actions">
      <div class="account-switcher">
        <span class="account-label">瑜版挸澧犵拹锔藉煕</span>
        <strong class="current-account" id="currentAccountLabel">local account</strong>
        <select id="accountSelect" title="閸掑洦宕茬拹锔藉煕"></select>
        <button type="button" id="createAccount">閺傛澘缂?/button>
        <button type="button" id="forgetAccount">缁夊娅?/button>
      </div>
      <button type="button" id="refresh">閸掗攱鏌婇悩鑸碘偓?/button>
    </div>
  </header>
  <main>
    <section class="chat-section">
      <div class="chat-shell">
        <aside class="chat-nav">
          <div class="brand-row">
            <div class="brand-dot">E</div>
            <div>Echo 閺呴缚鍏樻担?/div>
          </div>
          <button class="nav-button primary" id="openSession">閿?閺傛澘顕拠?/button>
          <div class="nav-list">
            <button class="nav-item active" id="chatNav">閳?鐎电鐦?/button>
            <button class="nav-item" id="memoryNav">閳?EchoMemory</button>
            <button class="nav-item" id="locomoNav">閳?LoCoMo 鐠囧嫭绁?/button>
            <button class="nav-item" id="graphEvalNav">閳?Graph 鐠囧嫭绁?/button>
          </div>
          <div class="nav-section-title">閺堚偓鏉╂垳绱扮拠?/div>
          <div id="sessionList"></div>
          <div class="chat-nav-footer">
            Alibaba / Qwen 瀹稿弶甯撮崗?br />
            EchoMemory 閸愭瑥鍙嗘稉搴㈩梾缁便垹鐤勯弮鍓佹晸閺?
          </div>
        </aside>
        <div class="chat-main" id="chatMain">
          <div class="main-view" id="chatMainView">
          <div>
            <div class="chat-topbar">
              <div class="chat-title">
                <span>EchoMemory Agent</span>
                <span class="model-pill" id="modelBadge">濡€崇€烽崝鐘烘祰娑?/span>
              </div>
            </div>
            <div class="identity-row">
              <label>agent_id<input id="agentId" value="demo-agent" /></label>
              <input id="userId" type="hidden" value="local" />
              <input id="sessionId" type="hidden" />
            </div>
          </div>
          <div class="chat-scroll" id="chatScroll">
            <div class="welcome" id="welcome">
              <div>
                <h2>娴犲﹤銇夐幆瀹犱喊閻愰€涚矆娑斿牞绱?/h2>
                <p>閸欐垼鎹ｆ稉鈧潪顔炬埂鐎?Agent 鐎电鐦介敍灞剧Х閹垯绱伴崘娆忓弳 EchoMemory 楠炴儼鐨熼悽?Alibaba 濡€崇€烽妴?/p>
                <div class="quick-prompts">
                  <button class="quickPrompt">鐢喗鍨滈弫瀵告倞 D03 閻ㄥ嫭褰佹禍銈嗘煙濡?/button>
                  <button class="quickPrompt">閹崵绮ㄨぐ鎾冲娴兼俺鐦芥稉顓犳畱閸忔娊鏁拋鏉跨箓</button>
                  <button class="quickPrompt">閻㈢喐鍨氭稉鈧稉顏冪瑏濮濄儲澧界悰宀冾吀閸?/button>
                </div>
              </div>
            </div>
            <div class="messages" id="messages"></div>
          </div>
          <div>
            <div class="composer-wrap">
              <div class="composer">
                <textarea id="userText" placeholder="缂?EchoMemory Agent 閸欐垿鈧焦绉烽幁?>鐢喗鍨滈弫瀵告倞娑撯偓娑?D03 閻ㄥ嫭褰佹禍銈嗘煙濡?/textarea>
                <div class="composer-actions">
                  <div class="tool-row">
                    <button class="chip-button" id="memoryChip">鐠佹澘绻傚Λ鈧槐?/button>
                    <label class="context-toggle" title="閹貉冨煑濡€崇€锋稉濠佺瑓閺傚洦妲搁崥锕€瀵橀崥顐ョ箮閺堢喎顕拠?>
                      <input id="includeHistory" type="checkbox" checked />
                      Conversation Tail
                    </label>
                    <select class="memory-test-select" id="memoryTestKind" title="闁瀚ㄧ拋鏉跨箓濞村鐦崷鐑樻珯">
                      <option value="all">閸忋劑鍎寸猾璇茬€风拋鏉跨箓</option>
                      <option value="profile">profile 閻劍鍩涢悽璇插剼</option>
                      <option value="preference">preference 閸嬪繐銈?/option>
                      <option value="entity">entity 鐎圭偘缍?/option>
                      <option value="event">event 娴滃娆?/option>
                      <option value="agent_case">agent_case 濡楀牅绶?/option>
                      <option value="pattern">pattern 濡€崇础</option>
                      <option value="tool_experience">tool_experience 瀹搞儱鍙跨紒蹇涚崣</option>
                    </select>
                    <button class="chip-button" id="memoryTestChip">鐠佹澘绻傚ù瀣槸</button>
                    <button class="chip-button" id="commitChip">閹绘劒姘﹁ぐ鎺撱€?/button>
                  </div>
                  <button class="send-button" id="sendChat" title="閸欐垿鈧礁鑻熼悽鐔稿灇">閳?/button>
                </div>
              </div>
            </div>
          </div>
          </div>
          <div class="memory-main" id="memoryMainView">
            <div class="chat-topbar">
              <div class="chat-title">
                <span>EchoMemory</span>
                <span class="model-pill">鐠佹澘绻傛潻鎰攽閺?/span>
              </div>
            </div>
            <div class="memory-page">
              <div class="memory-hero">
                <h2>EchoMemory 閻樿埖鈧?/h2>
                <p>鏉╂瑩鍣烽懕姘値瑜版挸澧?Agent 娴ｈ法鏁ら惃鍕唶韫囧棝鍘ょ純顔衡偓涔籩ssion 閺傚洣娆㈤弽鎴欌偓浣风皑娴犺泛鎷伴張鈧崥搴濈濞嗏剝甯撮崣锝呮惙鎼存柣鈧倸顕拠婵堢崶閸欙絽褰ч崷銊ｂ偓灞筋嚠鐠囨縿鈧秹銆夐弰鍓с仛閵?/p>
              </div>
              <div class="memory-grid">
                <div class="memory-card">
                  <strong>瑜版挸澧犳导姘崇樈</strong>
                  <span id="memorySessionLabel">-</span>
                </div>
                <div class="memory-card">
                  <strong>濡€崇€?/strong>
                  <span id="memoryModelLabel">-</span>
                </div>
                <div class="memory-card">
                  <strong>EchoMemory Target</strong>
                  <span id="memoryTargetLabel">echo://sessions/-</span>
                </div>
                <div class="memory-card">
                  <strong>瑜版帗銆傞悩鑸碘偓?/strong>
                  <span id="commitStatusLabel">閺堫亝褰佹禍?/span>
                </div>
                <div class="memory-card">
                  <strong>閹垮秳缍?/strong>
                  <button class="chip-button" id="memoryRefresh">閸掗攱鏌?EchoMemory 娣団剝浼?/button>
                </div>
              </div>
              <div class="commit-memory-detail" id="commitMemorySummary">
                <div class="commit-memory-empty">鐎瑰本鍨氶幓鎰唉閸氬函绱濇潻娆撳櫡娴兼碍妯夌粈鐑樻拱濞?commit 鐎圭偤妾幎钘夊絿閻ㄥ嫯顔囪箛鍡欒閸ㄥ鎷扮拠锔藉剰閵?/div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
    <section>
      <div class="panel-head">
        <h2 id="sideTitle">娑撳﹣绗呴弬鍥梾閺屻儱娅?/h2>
        <span class="status" id="sideStatus">濡€崇€锋稉濠佺瑓閺?/span>
      </div>
      <div class="panel-body inspect-grid">
        <div class="side-view active" id="contextPanel">
          <label>缂佸嫯顥婇崥搴ｆ畱濡€崇€锋稉濠佺瑓閺?/label>
          <div id="contextView" class="context-view">閸欐垿鈧焦绉烽幁顖氭倵閿涘矁绻栭柌灞肩窗閺勫墽銇氶張顒冪枂鐎圭偤妾幏鍏煎复楠炶泛褰傞柅浣虹舶濡€崇€烽惃鍕瑐娑撳鏋冮妴?/div>
        </div>
        <div class="side-view" id="memoryPanel">
          <label>閺呴缚鍏樻担鎾诲帳缂?/label>
          <pre id="config">{}</pre>
          <label>EchoMemory Runtime</label>
          <pre id="runtime">{}</pre>
          <label>閺傚洣娆㈢化鑽ょ埠閻╊喗鐖?/label>
          <div class="fs-controls">
            <label>閻╊喗鐖?URI<input id="fsTarget" value="echo://sessions/chat-001" /></label>
            <label>濞ｅ崬瀹?input id="fsDepth" value="5" /></label>
          </div>
          <div class="fs-actions">
            <button id="treeMode">閺嶆垹濮哥仦鏇炵磻</button>
            <button id="flatMode">楠炴娊鎽电仦鏇炵磻</button>
            <button id="refreshTree">閸掗攱鏌婇惄顔界垼</button>
            <button id="treeEngineTarget">Tree 瀵洘鎼?/button>
          </div>
          <div class="fs-view" id="tree"></div>
          <label>Selected File</label>
          <pre id="fileContent">閻愮懓鍤弬鍥︽閺屻儳婀呴崘鍛啇</pre>
          <label>Events</label>
          <pre id="events">{}</pre>
          <label>Commit Memory Details</label>
          <div class="commit-memory-detail" id="commitMemoryDetails">
            <div class="commit-memory-empty">閺嗗倹妫?commit 鐠佹澘绻傜拠锔藉剰閵?/div>
          </div>
          <label>Last Response</label>
          <pre id="last">{}</pre>
        </div>
      </div>
    </section>
  </main>
  <div class="modal-backdrop" id="accountPromptModal" aria-hidden="true">
    <div class="modal-card" role="dialog" aria-modal="true" aria-labelledby="accountPromptTitle">
      <h3 id="accountPromptTitle">閺傛澘缂?/h3>
      <p>鏉堟挸鍙嗘稉鈧稉顏冪┒娴滃氦鐦戦崚顐ゆ畱鐠愶附鍩涢崥宥囆為妴?/p>
      <label>鐠愶附鍩涢崥宥囆?input id="accountPromptInput" autocomplete="off" /></label>
      <div class="modal-actions">
        <button type="button" id="accountPromptCancel">閸欐牗绉?/button>
        <button type="button" id="accountPromptConfirm">绾喖鐣?/button>
      </div>
    </div>
  </div>
  <script>
    const $ = (id) => document.getElementById(id);
    let currentTree = {uri: "", entries: []};
    let fsMode = "tree";
    let sessions = [];
    let activeSession = null;
    let lastCommitMemorySummary = null;
    let accounts = [];
    let activeAccount = null;
    let accountRevision = 0;
    const memoryTestScenarios = {
      profile: ["閹存垶妲搁崥搴ｎ伂瀹搞儳鈻肩敮鍫礉闂€鎸庢埂鐠愮喕鐭?EchoMemory 鐠佹澘绻傜化鑽ょ埠閻ㄥ嫬鎮楃粩顖涚仸閺嬪嫨鈧椒绱扮拠婵嗙秺濡楋絽鎷扮拋鏉跨箓閹惰棄褰囬柧鎹愮熅閵?],
      preference: ["閹存垵鏋╁▎顫稑娴犮儱鎮楁妯款吇閸忓牏绮扮紒鎾诡啈閿涘苯鍟€缂佹瑧鐣濆ú浣诡劄妤犮倧绱辨俊鍌涚亯闂団偓鐟曚礁鐫嶅鈧敍灞藉晙鐞涖儱鍘栭崗鎶芥暛閸樼喎娲滈崪宀勭崣鐠囦焦鏌熷蹇嬧偓?],
      entity: ["EchoMemory 妞ゅ湱娲伴柌宀€娈?TreeMemoryEngine 閺勵垳绮ㄩ弸鍕闂€鎸庢埂鐠佹澘绻傚鏇熸惛閿涘矁绀嬬拹锝呮躬 session commit 閸氬氦顕伴崣鏍︾窗鐠囨繂缍婂锝忕礉楠炲墎鏁撻幋?profile閵嗕垢reference閵嗕躬ntity閵嗕躬vent閵嗕恭ase閵嗕垢attern 閸?tool 閻╃鍙х拋鏉跨箓閵?],
      event: ["娴犲﹤銇夐崘鍐茬暰閹?EchoMemory 閻ㄥ嫬褰崶鐐扮喘閸栨牗鏂侀崚棰佺瑓娑撯偓闂冭埖顔岄敍娑欐拱闂冭埖顔屾导妯哄帥鐎瑰本鍨?TreeMemoryEngine 閻?commit 閹惰棄褰囨宀冪槈閿涘瞼鈥樼拋?event 缁鐎烽崣顖欎簰娴犲簼绱扮拠婵嗙秺濡楋絿鏁撻幋鎰┾偓?],
      agent_case: ["鏉╂瑤閲滄禒璇插闁洤鍩岄惃鍕６妫版ɑ妲?commit 閸氬酣銆夐棃銏犲涧閺勫墽銇氬▽鈩冩箒閹惰棄褰囩拋鏉跨箓閿涙稑甯崶鐘虫Ц濞村鐦崣銉ョ摍娣団剝浼呮径顏勬€ユ稉鏂跨秼閸撳秷绻嶇悰宀€娈戦弰?tree 瀵洘鎼搁敍娑溞掗崘鍐插濞夋洘妲搁幎濠冪ゴ鐠囨洖顕拠婵囨暭閹存劕瀵橀崥顐ｆ绾喕绗傛稉瀣瀮閵嗕礁甯崶鐘叉嫲缂佹挻鐏夐惃鍕彯鐠愩劑鍣洪弽铚傜伐閿涙稓绮ㄩ弸婊勬Ц commit 閹芥顩﹂懗鑺ユ▔缁€鐑樻拱鏉烆喗濞婇崣鏍у毉閻ㄥ嫯顔囪箛鍡欒閸ㄥ鈧?],
      pattern: ["濮ｅ繑顐兼穱顔芥暭閸忣剙绱戦幒銉ュ經閹存牞顔囪箛鍡樺▕閸欐牞顫夐崚娆忔倵閿涘苯娴愮€规碍绁︾粙瀣Ц閸忓牐藟閸忓懘鎷＄€佃鈧呮畱閸楁洖鍘撳ù瀣槸閿涘苯鍟€鐠烘垹娴夐崗铏ゴ鐠囨洖顨滄禒璁圭礉閺堚偓閸氬酣鍣搁崥顖涙箛閸斺€宠嫙閻劑銆夐棃顫瑐閻ㄥ嫯顔囪箛鍡樼ゴ鐠囨洜鈥樼拋?commit 缂佹挻鐏夐妴?],
      tool_experience: ["[ToolCall] rg -n \"閺堫剚顐?commit 濞屸剝婀侀幎钘夊絿閸戞椽鏆遍張鐔活唶韫囧敗memory_kinds\" src e:/echomem-workspace/echomemory\n瀹搞儱鍙跨紒蹇涚崣閿涙矮濞囬悽?rg 鐎规矮缍呯拋鏉跨箓閹惰棄褰囬梻顕€顣介弮璁圭礉閸忓牊鐓＄划鍓р€橀幓鎰仛閺傚洦顢嶉崪灞藉彠闁款喖鐡у▓纰夌幢婵″倹鐏夌捄顖氱窞閼煎啫娲挎径顏勩亣鐎佃壈鍤х紒鎾寸亯閸ｎ亜锛愭姗堢礉鐏忚鲸鏁圭粣鍕煂 src/agent/server.py閵嗕辜ext_processor.py 閸?session_service.py閵?],
      all: [
        "閹存垶妲搁崥搴ｎ伂瀹搞儳鈻肩敮鍫礉闂€鎸庢埂鐠愮喕鐭?EchoMemory 鐠佹澘绻傜化鑽ょ埠閻ㄥ嫬鎮楃粩顖涚仸閺嬪嫨鈧椒绱扮拠婵嗙秺濡楋絽鎷扮拋鏉跨箓閹惰棄褰囬柧鎹愮熅閵?,
        "閹存垵鏋╁▎顫稑娴犮儱鎮楁妯款吇閸忓牏绮扮紒鎾诡啈閿涘苯鍟€缂佹瑧鐣濆ú浣诡劄妤犮倧绱辨俊鍌涚亯闂団偓鐟曚礁鐫嶅鈧敍灞藉晙鐞涖儱鍘栭崗鎶芥暛閸樼喎娲滈崪宀勭崣鐠囦焦鏌熷蹇嬧偓?,
        "EchoMemory 妞ゅ湱娲伴柌宀€娈?TreeMemoryEngine 閺勵垳绮ㄩ弸鍕闂€鎸庢埂鐠佹澘绻傚鏇熸惛閿涘矁绀嬬拹锝呮躬 session commit 閸氬氦顕伴崣鏍︾窗鐠囨繂缍婂锝忕礉楠炲墎鏁撻幋?profile閵嗕垢reference閵嗕躬ntity閵嗕躬vent閵嗕恭ase閵嗕垢attern 閸?tool 閻╃鍙х拋鏉跨箓閵?,
        "娴犲﹤銇夐崘鍐茬暰閹?EchoMemory 閻ㄥ嫬褰崶鐐扮喘閸栨牗鏂侀崚棰佺瑓娑撯偓闂冭埖顔岄敍娑欐拱闂冭埖顔屾导妯哄帥鐎瑰本鍨?TreeMemoryEngine 閻?commit 閹惰棄褰囨宀冪槈閿涘瞼鈥樼拋?event 缁鐎烽崣顖欎簰娴犲簼绱扮拠婵嗙秺濡楋絿鏁撻幋鎰┾偓?,
        "鏉╂瑤閲滄禒璇插闁洤鍩岄惃鍕６妫版ɑ妲?commit 閸氬酣銆夐棃銏犲涧閺勫墽銇氬▽鈩冩箒閹惰棄褰囩拋鏉跨箓閿涙稑甯崶鐘虫Ц濞村鐦崣銉ョ摍娣団剝浼呮径顏勬€ユ稉鏂跨秼閸撳秷绻嶇悰宀€娈戦弰?tree 瀵洘鎼搁敍娑溞掗崘鍐插濞夋洘妲搁幎濠冪ゴ鐠囨洖顕拠婵囨暭閹存劕瀵橀崥顐ｆ绾喕绗傛稉瀣瀮閵嗕礁甯崶鐘叉嫲缂佹挻鐏夐惃鍕彯鐠愩劑鍣洪弽铚傜伐閿涙稓绮ㄩ弸婊勬Ц commit 閹芥顩﹂懗鑺ユ▔缁€鐑樻拱鏉烆喗濞婇崣鏍у毉閻ㄥ嫯顔囪箛鍡欒閸ㄥ鈧?,
        "濮ｅ繑顐兼穱顔芥暭閸忣剙绱戦幒銉ュ經閹存牞顔囪箛鍡樺▕閸欐牞顫夐崚娆忔倵閿涘苯娴愮€规碍绁︾粙瀣Ц閸忓牐藟閸忓懘鎷＄€佃鈧呮畱閸楁洖鍘撳ù瀣槸閿涘苯鍟€鐠烘垹娴夐崗铏ゴ鐠囨洖顨滄禒璁圭礉閺堚偓閸氬酣鍣搁崥顖涙箛閸斺€宠嫙閻劑銆夐棃顫瑐閻ㄥ嫯顔囪箛鍡樼ゴ鐠囨洜鈥樼拋?commit 缂佹挻鐏夐妴?,
        "[ToolCall] rg -n \"閺堫剚顐?commit 濞屸剝婀侀幎钘夊絿閸戞椽鏆遍張鐔活唶韫囧敗memory_kinds\" src e:/echomem-workspace/echomemory\n瀹搞儱鍙跨紒蹇涚崣閿涙矮濞囬悽?rg 鐎规矮缍呯拋鏉跨箓閹惰棄褰囬梻顕€顣介弮璁圭礉閸忓牊鐓＄划鍓р€橀幓鎰仛閺傚洦顢嶉崪灞藉彠闁款喖鐡у▓纰夌幢婵″倹鐏夌捄顖氱窞閼煎啫娲挎径顏勩亣鐎佃壈鍤х紒鎾寸亯閸ｎ亜锛愭姗堢礉鐏忚鲸鏁圭粣鍕煂 src/agent/server.py閵嗕辜ext_processor.py 閸?session_service.py閵?
      ]
    };
    const sessionStoreKey = "echomem-agent.sessions.v1";
    const accountStoreKey = "echomem-agent.accounts.v1";

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
    function setCommitStatus(text, detail = null) {
      $("commitStatusLabel").textContent = text;
      $("sideStatus").textContent = text;
      if (detail) show("last", detail);
    }
    function authHeaders(base = {}) {
      const headers = {...base};
      if (activeAccount?.authKey) headers["X-Auth-Key"] = activeAccount.authKey;
      return headers;
    }
    function isCurrentAccount(revision) {
      return revision === accountRevision;
    }
    function accountSessionStoreKey() {
      return `${sessionStoreKey}.${activeAccount?.id || "local"}`;
    }
    function sessionStoreKeyFor(accountId) {
      return `${sessionStoreKey}.${accountId || "local"}`;
    }
    function loadAccounts() {
      try {
        const parsed = JSON.parse(localStorage.getItem(accountStoreKey) || "[]");
        accounts = Array.isArray(parsed) ? parsed : [];
      } catch {
        accounts = [];
      }
      if (accounts.length === 0) accounts = [{id: "local", label: "local account", authKey: ""}];
      const activeId = localStorage.getItem(`${accountStoreKey}.active`) || "";
      activeAccount = accounts.find((item) => item.id === activeId) || accounts[0];
    }
    function saveAccounts() {
      localStorage.setItem(accountStoreKey, JSON.stringify(accounts.slice(0, 20)));
      localStorage.setItem(`${accountStoreKey}.active`, activeAccount?.id || "local");
    }
    function renderAccounts() {
      const select = $("accountSelect");
      select.innerHTML = "";
      for (const account of accounts) {
        const option = document.createElement("option");
        option.value = account.id;
        option.textContent = account.label || account.id;
        option.selected = account.id === activeAccount?.id;
        select.appendChild(option);
      }
      $("currentAccountLabel").textContent = activeAccount?.label || "local account";
      $("userId").value = activeAccount?.id || "local";
      updateAccountActions();
    }
    function updateAccountActions() {
      const isLocalAccount = !activeAccount || activeAccount.id === "local";
      $("forgetAccount").disabled = isLocalAccount;
      $("forgetAccount").title = isLocalAccount ? "local account 娑撳秷鍏樼粔濠氭珟" : "缁夊娅庤ぐ鎾冲鐠愶附鍩?;
    }
    let accountPromptResolver = null;
    function closeAccountPrompt(value) {
      const resolver = accountPromptResolver;
      accountPromptResolver = null;
      $("accountPromptModal").classList.remove("open");
      $("accountPromptModal").setAttribute("aria-hidden", "true");
      $("accountPromptInput").value = "";
      if (resolver) resolver(value);
    }
    function openAccountPrompt(title, defaultValue) {
      $("accountPromptTitle").textContent = title;
      $("accountPromptInput").value = defaultValue;
      $("accountPromptModal").classList.add("open");
      $("accountPromptModal").setAttribute("aria-hidden", "false");
      return new Promise((resolve) => {
        accountPromptResolver = resolve;
        queueMicrotask(() => {
          $("accountPromptInput").focus();
          $("accountPromptInput").select();
        });
      });
    }
    async function accountRequest(path, payload) {
      const response = await fetch(path, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload || {})
      });
      const data = await response.json();
      show("last", data);
      if (!response.ok || data.error) throw new Error(data.message || data.error || response.statusText);
      return data;
    }
    function isInvalidAuthKeyError(data) {
      const message = `${data?.message || ""} ${data?.error || ""}`;
      return message.includes("invalid X-Auth-Key") || message.includes("echomemory_http_401");
    }
    async function refreshActiveAccount() {
      if (!activeAccount || activeAccount.id === "local") throw new Error("current account has no refreshable auth key");
      await ensureNamedAccount(activeAccount.label || "locomo", {activate: true});
    }
    function switchAccount(accountId) {
      accountRevision += 1;
      activeAccount = accounts.find((item) => item.id === accountId) || accounts[0];
      saveAccounts();
      clearAccountView();
      loadSessions();
      if (sessions.length > 0) setActiveSession(sessions[0]);
      else resetConversation();
      renderAccounts();
      refreshInspectors();
    }
    function clearAccountView() {
      sessions = [];
      activeSession = null;
      lastCommitMemorySummary = null;
      $("messages").innerHTML = "";
      $("contextView").textContent = "閸掑洦宕茬拹锔藉煕閸氬函绱濈亸鍡楀涧閺勫墽銇氳ぐ鎾冲鐠愶附鍩涙稉瀣畱娴兼俺鐦芥稉濠佺瑓閺傚洢鈧?;
      $("tree").innerHTML = "";
      $("fileContent").textContent = "閻愮懓鍤弬鍥︽閺屻儳婀呴崘鍛啇";
      show("events", {});
      renderCommitMemoryDetails(null);
      renderSessions();
    }
    async function createAccount() {
      const label = (await openAccountPrompt("閺傛澘缂?, `account ${accounts.filter((item) => item.id !== "local").length + 1}`) || "").trim();
      if (!label) return;
      $("status").textContent = `濮濓絽婀弬鏉跨紦鐠愶附鍩?${label}...`;
      $("createAccount").disabled = true;
      try {
        const data = await accountRequest("/agent/accounts/create", {label});
        const account = data.account;
        if (!account?.authKey) throw new Error("account creation returned no authKey");
        activeAccount = {
          id: account.id,
          label: account.label,
          tenantId: account.tenantId,
          userId: account.userId,
          authKey: account.authKey
        };
        accounts = [activeAccount, ...accounts];
        accountRevision += 1;
        saveAccounts();
        clearAccountView();
        renderAccounts();
        resetConversation();
        $("status").textContent = `瀹稿弶鏌婂鍝勮嫙閸掑洦宕查崚鎷屽閹?${activeAccount.label || activeAccount.id}`;
        show("last", {account: label, status: "created"});
      } finally {
        $("createAccount").disabled = false;
      }
    }
    async function ensureNamedAccount(label = "locomo", options = {}) {
      const data = await accountRequest("/agent/accounts/ensure", {label});
      const account = data.account;
      if (!account?.authKey) throw new Error("account ensure returned no authKey");
      const existing = accounts.find((item) => item.id === account.id || item.label === account.label);
      const ensuredAccount = {
        id: account.id,
        label: account.label,
        tenantId: account.tenantId,
        userId: account.userId,
        authKey: account.authKey
      };
      if (options.activate !== false) activeAccount = ensuredAccount;
      accounts = existing
        ? [ensuredAccount, ...accounts.filter((item) => item !== existing)]
        : [ensuredAccount, ...accounts];
      if (options.activate !== false) accountRevision += 1;
      saveAccounts();
      if (options.activate !== false) clearAccountView();
      renderAccounts();
      if (options.activate !== false) resetConversation();
      show("last", {account: ensuredAccount.label, status: data.created ? "created" : "found"});
    }
    async function ensureLocomoAccountOnStartup() {
      if (accounts.some((item) => item.label === "locomo" || item.id === "locomo")) return;
      try {
        await ensureNamedAccount("locomo", {activate: false});
      } catch (error) {
        show("last", {warning: `locomo account ensure failed: ${error.message}`});
      }
    }
    async function forgetAccount() {
      if (!activeAccount) return;
      if (activeAccount.id === "local") {
        show("last", {warning: "local account 娑撳秷鍏樼粔濠氭珟閿涘矁顕崗鍫濆瀼閹广垹鍩岄崗鏈电铂鐠愶附鍩涢妴?});
        return;
      }
      const removedAccount = activeAccount;
      $("forgetAccount").disabled = true;
      try {
        try {
          await fetch("/api/auth/account/delete", {method: "POST", headers: authHeaders({"Content-Type": "application/json"}), body: "{}"});
        } catch {
          // Local removal should still work if EchoMemory is temporarily unavailable.
        }
        try {
          if (removedAccount.id !== "local") {
            await accountRequest("/agent/accounts/forget", {id: removedAccount.id, label: removedAccount.label});
          }
        } catch {
          // Registry cleanup is best-effort; the local account list is the source for this page.
        }
        localStorage.removeItem(`${sessionStoreKey}.${removedAccount.id}`);
        accounts = removedAccount.id === "local"
          ? [{id: "local", label: "local account", authKey: ""}, ...accounts.filter((item) => item.id !== "local")]
          : accounts.filter((item) => item.id !== removedAccount.id);
        activeAccount = accounts[0] || {id: "local", label: "local account", authKey: ""};
        accountRevision += 1;
        saveAccounts();
        switchAccount(activeAccount.id);
        show("last", {account: removedAccount.label || removedAccount.id, status: "removed"});
      } finally {
        $("forgetAccount").disabled = false;
      }
    }
    function findSession(id) {
      return sessions.find((item) => item.id === id) || null;
    }
    function loadSessions() {
      try {
        const parsed = JSON.parse(localStorage.getItem(accountSessionStoreKey()) || "[]");
        sessions = Array.isArray(parsed) ? parsed : [];
      } catch {
        sessions = [];
      }
    }
    function saveSessions() {
      localStorage.setItem(accountSessionStoreKey(), JSON.stringify(sessions.slice(0, 30)));
    }
    function loadSessionsForAccount(accountId) {
      try {
        const parsed = JSON.parse(localStorage.getItem(sessionStoreKeyFor(accountId)) || "[]");
        return Array.isArray(parsed) ? parsed : [];
      } catch {
        return [];
      }
    }
    function saveSessionsForAccount(accountId, accountSessions) {
      localStorage.setItem(sessionStoreKeyFor(accountId), JSON.stringify(accountSessions.slice(0, 30)));
    }
    function rememberMessageForAccount(accountId, targetSessionId, role, content) {
      const accountSessions = loadSessionsForAccount(accountId);
      const session = accountSessions.find((item) => item.id === targetSessionId);
      if (!session) return;
      session.messages = Array.isArray(session.messages) ? session.messages : [];
      session.messages.push({role, content});
      session.updatedAt = new Date().toISOString();
      saveSessionsForAccount(accountId, [session, ...accountSessions.filter((item) => item.id !== session.id)]);
    }
    function rememberContextForAccount(accountId, targetSessionId, messages, trace = null) {
      const accountSessions = loadSessionsForAccount(accountId);
      const session = accountSessions.find((item) => item.id === targetSessionId);
      if (!session) return;
      session.context = messages;
      session.contextTrace = trace;
      session.updatedAt = new Date().toISOString();
      saveSessionsForAccount(accountId, [session, ...accountSessions.filter((item) => item.id !== session.id)]);
    }
    function createConversation() {
      activeSession = {
        id: createSessionId(),
        title: "閺傛澘顕拠?,
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
        empty.innerHTML = `<span class="recent-title">閺嗗倹妫ゆ导姘崇樈</span>`;
        list.appendChild(empty);
        return;
      }
      for (const session of sessions) {
        const row = document.createElement("div");
        row.className = `recent-chat ${activeSession?.id === session.id ? "active" : ""}`;
        row.innerHTML = `
          <span class="recent-title">${escapeHtml(session.title || session.id)}</span>
          <button class="delete-session" title="閸掔娀娅庢导姘崇樈">鑴?/button>
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
      else $("contextView").textContent = "閸欐垿鈧焦绉烽幁顖氭倵閿涘矁绻栭柌灞肩窗閺勫墽銇氶張顒冪枂鐎圭偤妾幏鍏煎复楠炶泛褰傞柅浣虹舶濡€崇€烽惃鍕瑐娑撳鏋冮妴?;
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
        if (layer.name !== "鏉╂垶婀＄€电鐦? || !Array.isArray(layer.message_indexes) || layer.message_indexes.length <= 1) continue;
        for (const index of layer.message_indexes) grouped.add(index);
        rows.push({
          title: `#${layer.message_indexes[0] + 1}-${layer.message_indexes[layer.message_indexes.length - 1] + 1} 鏉╂垶婀＄€电鐦絗,
          role: "conversation",
          layer,
          content: layer.message_indexes.map((index) => {
            const message = messages[index] || {};
            const speaker = message.role === "assistant" ? "閸斺晜澧? : "閻劍鍩?;
            return `閵?{speaker}閵嗘叚n${message.content || ""}`;
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
      if (role === "system") return "缁崵绮?;
      if (role === "user") return "閻劍鍩?;
      if (role === "assistant") return "閸斺晜澧?;
      if (role === "tool") return "瀹搞儱鍙?;
      return role || "濞戝牊浼?;
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
      $("contextView").textContent = "閺傛澘顕拠婵嗗嚒閸掓稑缂撻妴鍌氬絺闁焦绉烽幁顖氭倵閿涘矁绻栭柌灞肩窗閺勫墽銇氶張顒冪枂鐎圭偤妾幏鍏煎复楠炶泛褰傞柅浣虹舶濡€崇€烽惃鍕瑐娑撳鏋冮妴?;
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
      $("sideTitle").textContent = isMemory ? "EchoMemory" : "娑撳﹣绗呴弬鍥梾閺屻儱娅?;
      $("sideStatus").textContent = isMemory ? "鐠佹澘绻傛稉搴ょ殶鐠囨洑淇婇幁? : "濡€崇€锋稉濠佺瑓閺?;
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
        container.innerHTML = `<div class="fs-empty">閻╊喗鐖ｆ稉瀣畯閺冪姴鍞寸€?/div>`;
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
          <div class="fs-meta">${entry.kind}${entry.kind === "file" ? " 璺?" + entry.size + "B" : ""}</div>
        `;
        if (entry.kind === "file") {
          row.onclick = () => readFile(entry.uri);
          row.title = "閻愮懓鍤拠璇插絿閺傚洣娆?;
        }
        container.appendChild(row);
      }
    }
    async function refreshTree() {
      const revision = accountRevision;
      try {
        const uri = $("fsTarget").value.trim();
        const depth = $("fsDepth").value.trim() || "5";
        currentTree = await fetch(`/agent/inspect/fs/tree?uri=${encodeURIComponent(uri)}&max_depth=${encodeURIComponent(depth)}`, {
          headers: authHeaders()
        }).then((r) => r.json());
        if (!isCurrentAccount(revision)) return;
        if (currentTree.error) throw new Error(currentTree.message || currentTree.error);
        renderFs(currentTree.entries);
      } catch (error) {
        if (!isCurrentAccount(revision)) return;
        $("tree").innerHTML = `<div class="fs-empty error">${escapeHtml(error.message)}</div>`;
      }
    }
    async function readFile(uri) {
      const revision = accountRevision;
      try {
        const data = await fetch(`/agent/inspect/fs/read?uri=${encodeURIComponent(uri)}`, {
          headers: authHeaders()
        }).then((r) => r.json());
        if (!isCurrentAccount(revision)) return;
        if (data.error) throw new Error(data.message || data.error);
        $("fileContent").textContent = data.text;
        show("last", data);
      } catch (error) {
        if (!isCurrentAccount(revision)) return;
        $("fileContent").textContent = error.message;
      }
    }
    async function request(path, options = {}) {
      const revision = accountRevision;
      const retriedAuth = Boolean(options.retriedAuth);
      const fetchOptions = {...options};
      delete fetchOptions.retriedAuth;
      let response;
      try {
        response = await fetch(path, {
          headers: authHeaders({"Content-Type": "application/json"}),
          ...fetchOptions
        });
      } catch (error) {
        throw new Error(`閺冪姵纭舵潻鐐村复 Agent 閺堝秴濮熼敍?{error.message}`);
      }
      const text = await response.text();
      let data = {};
      try {
        data = text ? JSON.parse(text) : {};
      } catch (error) {
        data = {error: "invalid_json_response", message: text || response.statusText};
      }
      if (isCurrentAccount(revision)) show("last", data);
      if (!response.ok && isCurrentAccount(revision) && isInvalidAuthKeyError(data) && !retriedAuth) {
        await refreshActiveAccount();
        return request(path, {...fetchOptions, retriedAuth: true});
      }
      if (!response.ok) throw new Error(data.message || data.error || response.statusText);
      if (isCurrentAccount(revision)) await refreshInspectors();
      return data;
    }
    async function waitCommit(sessionIdForCommit, archiveId) {
      const revision = accountRevision;
      for (let attempt = 0; attempt < 1200; attempt += 1) {
        const response = await fetch(`/api/sessions/${encodeURIComponent(sessionIdForCommit)}/commits/${encodeURIComponent(archiveId)}`, {
          headers: authHeaders()
        });
        const data = await response.json();
        if (!isCurrentAccount(revision)) throw new Error("account switched");
        show("last", data);
        if (!response.ok || data.error) throw new Error(data.message || data.error || `commit status ${response.status}`);
        const statusValue = data.status?.status || "processing";
        if (statusValue === "completed") {
          setCommitStatus(`瑜版帗銆傜€瑰本鍨氶敍?{archiveId}`, data);
          return data.status;
        }
        if (statusValue === "failed") {
          setCommitStatus(`瑜版帗銆傛径杈Е閿?{archiveId}`, data);
          throw new Error(data.status.error || "commit failed");
        }
        setCommitStatus(`瑜版帗銆傛径鍕倞娑擃叏绱?{archiveId}閿?{statusValue}閿涘ˇ, data);
        await new Promise((resolve) => setTimeout(resolve, 500));
      }
      throw new Error(`commit timeout: ${archiveId}`);
    }
    async function fetchCommitMemories(sessionIdForCommit, commitId) {
      const revision = accountRevision;
      const data = await fetch(`/api/sessions/${encodeURIComponent(sessionIdForCommit)}/commits/${encodeURIComponent(commitId)}/memories`, {
        headers: authHeaders()
      }).then((r) => r.json());
      if (!isCurrentAccount(revision)) throw new Error("account switched");
      show("last", data);
      if (data.error) throw new Error(data.message || data.error);
      return data.summary || {memory_kinds: [], memories: []};
    }
    function formatCommitMemories(summary) {
      const kinds = Array.isArray(summary.memory_kinds) ? summary.memory_kinds : [];
      if (kinds.length === 0) return "閺堫剚顐?commit 濞屸剝婀侀幎钘夊絿閸戞椽鏆遍張鐔活唶韫囧棎鈧?;
      const count = Array.isArray(summary.memories) ? summary.memories.length : 0;
      return `閺堫剚顐?commit 閹惰棄褰囨禍?${count} 閺壜ゎ唶韫囧棴绱濈猾璇茬€烽敍?{kinds.join(", ")}`;
    }
    function renderCommitMemoryDetails(summary) {
      lastCommitMemorySummary = summary;
      const html = commitMemoryHtml(summary);
      $("commitMemorySummary").innerHTML = html;
      $("commitMemoryDetails").innerHTML = html;
    }
    function commitMemoryHtml(summary) {
      if (!summary) return `<div class="commit-memory-empty">閺嗗倹妫?commit 鐠佹澘绻傜拠锔藉剰閵?/div>`;
      const kinds = Array.isArray(summary.memory_kinds) ? summary.memory_kinds : [];
      const memories = Array.isArray(summary.memories) ? summary.memories : [];
      if (kinds.length === 0) return `<div class="commit-memory-empty">閺堫剚顐?commit 濞屸剝婀侀幎钘夊絿閸戞椽鏆遍張鐔活唶韫囧棎鈧?/div>`;
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
      const revision = accountRevision;
      try {
        const config = await fetch("/agent/config").then((r) => r.json());
        if (!isCurrentAccount(revision)) return;
        show("config", config);
        $("modelBadge").textContent = `${config.model?.provider || "濡€崇€?} / ${config.model?.model || "閺堫亞鐓?}`;
        $("memoryModelLabel").textContent = `${config.model?.provider || "濡€崇€?} / ${config.model?.model || "閺堫亞鐓?}`;
        $("status").textContent = `濡€崇€?${config.model?.provider || ""} / EchoMemory ${sessionId()}`;
      } catch (error) {
        if (!isCurrentAccount(revision)) return;
        show("config", {error: error.message});
        $("modelBadge").textContent = "濡€崇€锋稉宥呭讲閻?;
        $("status").innerHTML = `<span class="error">${escapeHtml(error.message)}</span>`;
      }
      try {
        const runtime = await fetch("/agent/inspect/runtime", {headers: authHeaders()}).then((r) => r.json());
        if (!isCurrentAccount(revision)) return;
        show("runtime", runtime);
      } catch (error) {
        if (!isCurrentAccount(revision)) return;
        show("runtime", {error: error.message});
      }
      if (!isCurrentAccount(revision)) return;
      syncFsTarget();
      $("memorySessionLabel").textContent = sessionId() || "-";
      $("memoryTargetLabel").textContent = $("fsTarget").value || "echo://sessions/-";
      await refreshTree();
      try {
        const events = await fetch("/agent/inspect/events", {headers: authHeaders()}).then((r) => r.json());
        if (!isCurrentAccount(revision)) return;
        show("events", events);
      } catch (error) {
        if (!isCurrentAccount(revision)) return;
        show("events", {error: error.message});
      }
    }
    $("openSession").onclick = async () => {
      resetConversation();
    };
    async function sendChatMessage(content, options = {}) {
      const revision = accountRevision;
      const originAccountId = activeAccount?.id || "local";
      const activeSessionId = ensureSessionId();
      const targetSession = findSession(activeSessionId);
      if (targetSession && targetSession.title === "閺傛澘顕拠?) {
        targetSession.title = content;
        if (activeSession?.id === targetSession.id) activeSession = targetSession;
        saveSessions();
        renderSessions();
      }
      bubble("user", content, {persist: options.persistUser !== false});
      const pending = document.createElement("div");
      pending.className = "msg assistant";
      pending.innerHTML = `<div class="meta">閸斺晜澧?/div>閻㈢喐鍨氭稉?..`;
      $("messages").appendChild(pending);
      $("chatScroll").scrollTop = $("chatScroll").scrollHeight;
      const chatPayload = {
        user_id: $("userId").value.trim(),
        agent_id: $("agentId").value.trim(),
        session_id: activeSessionId,
        message: content,
        include_history: $("includeHistory").checked,
        stream: false
      };
      try {
        $("contextView").textContent = "濮濓絽婀紒鍕棅閺堫剝鐤嗗Ο鈥崇€锋稉濠佺瑓閺?..";
        const preview = await request("/agent/context", {
          method: "POST",
          body: JSON.stringify(chatPayload)
        });
        if (!isCurrentAccount(revision)) {
          rememberContextForAccount(originAccountId, activeSessionId, preview.messages, preview.context_trace);
        } else {
          rememberContextFor(activeSessionId, preview.messages, preview.context_trace);
          if (activeSession?.id === activeSessionId) showContext(preview.messages, preview.context_trace);
        }
        const data = await request("/agent/chat", {
          method: "POST",
          body: JSON.stringify(chatPayload)
        });
        if (!isCurrentAccount(revision)) {
          rememberContextForAccount(originAccountId, activeSessionId, data.messages, data.context_trace);
          rememberMessageForAccount(originAccountId, activeSessionId, "assistant", data.assistant.content);
          return data;
        }
        rememberContextFor(activeSessionId, data.messages, data.context_trace);
        rememberMessageFor(activeSessionId, "assistant", data.assistant.content);
        if (activeSession?.id === activeSessionId) {
          if (pending.isConnected) {
            pending.innerHTML = `<div class="meta">閸斺晜澧?/div>${escapeHtml(data.assistant.content)}`;
            showContext(data.messages, data.context_trace);
          } else {
            renderConversation();
          }
        }
        return data;
      } catch (error) {
        if (!isCurrentAccount(revision)) {
          rememberMessageForAccount(originAccountId, activeSessionId, "tool", `鐠囬攱鐪版径杈Е閿?{error.message}`);
          throw error;
        }
        rememberMessageFor(activeSessionId, "tool", `鐠囬攱鐪版径杈Е閿?{error.message}`);
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
      $("commitChip").disabled = true;
      setCommitStatus("濮濓絽婀幓鎰唉瑜版帗銆?..");
      if (activeSession?.id === sessionIdForCommit) bubble("tool", "濮濓絽婀幓鎰唉瑜版帗銆?..");
      const data = await request(`/api/sessions/${encodeURIComponent(sessionIdForCommit)}/commit`, {
        method: "POST",
        body: "{}"
      });
      const commitId = data.result.commit_id || data.result.archive_id;
      setCommitStatus(`瑜版帗銆傚鎻掑綀閻炲棴绱?{commitId}`, data);
      if (activeSession?.id === sessionIdForCommit) bubble("tool", `瑜版帗銆傚鎻掑綀閻炲棴绱?{commitId}`);
      else rememberMessageFor(sessionIdForCommit, "tool", `瑜版帗銆傚鎻掑綀閻炲棴绱?{commitId}`);
      try {
        const status = await waitCommit(sessionIdForCommit, commitId);
        const memorySummary = await fetchCommitMemories(sessionIdForCommit, commitId);
        renderCommitMemoryDetails(memorySummary);
        await refreshInspectors();
        setCommitStatus(`瑜版帗銆傜€瑰本鍨氶敍?{commitId}`, {status, summary: memorySummary});
        const message = `瑜版帗銆傜€瑰本鍨氶敍?{commitId}\n${formatCommitMemories(memorySummary)}`;
        if (activeSession?.id === sessionIdForCommit) bubble("tool", message);
        else rememberMessageFor(sessionIdForCommit, "tool", message);
        return memorySummary;
      } catch (error) {
        setCommitStatus(`瑜版帗銆傛径杈Е閿?{commitId}`, {error: error.message});
        if (activeSession?.id === sessionIdForCommit) bubble("tool", `瑜版帗銆傛径杈Е閿?{error.message}`);
        else rememberMessageFor(sessionIdForCommit, "tool", `瑜版帗銆傛径杈Е閿?{error.message}`);
        throw error;
      } finally {
        $("commitChip").disabled = false;
      }
    }
    async function runMemoryTest() {
      const kind = $("memoryTestKind").value || "all";
      const scenario = memoryTestScenarios[kind] || memoryTestScenarios.all;
      $("memoryTestChip").disabled = true;
      $("commitChip").disabled = true;
      try {
        setSideView("context");
        bubble("tool", `瀵偓婵顔囪箛鍡樼ゴ鐠囨洩绱?{kind}閿涘苯鍙?${scenario.length} 鏉烆喓鈧繖);
        for (const content of scenario) {
          await sendChatMessage(content);
        }
        bubble("tool", "濞村鐦€电鐦界€瑰本鍨氶敍灞界磻婵鍤滈崝銊﹀絹娴溿倕鑻熼幎钘夊絿鐠佹澘绻傞妴?);
        const summary = await commitCurrentSession();
        setSideView("memory");
        show("last", {memory_test: kind, summary});
      } catch (error) {
        bubble("tool", `鐠佹澘绻傚ù瀣槸婢惰精瑙﹂敍?{error.message}`);
      } finally {
        $("memoryTestChip").disabled = false;
        $("commitChip").disabled = false;
      }
    }
    $("refresh").onclick = refreshInspectors;
    $("accountSelect").onchange = () => switchAccount($("accountSelect").value);
    $("createAccount").onclick = async () => {
      try {
        await createAccount();
      } catch (error) {
        $("status").textContent = `閺傛澘缂撴径杈Е閿?{error.message}`;
        show("last", {error: error.message});
      }
    };
    $("forgetAccount").onclick = async () => {
      try {
        await forgetAccount();
      } catch (error) {
        show("last", {error: error.message});
      }
    };
    $("accountPromptConfirm").onclick = () => closeAccountPrompt($("accountPromptInput").value);
    $("accountPromptCancel").onclick = () => closeAccountPrompt("");
    $("accountPromptModal").onclick = (event) => {
      if (event.target === $("accountPromptModal")) closeAccountPrompt("");
    };
    $("accountPromptInput").addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        closeAccountPrompt($("accountPromptInput").value);
      } else if (event.key === "Escape") {
        event.preventDefault();
        closeAccountPrompt("");
      }
    });
    $("refreshTree").onclick = refreshTree;
    $("treeMode").onclick = () => { fsMode = "tree"; renderFs(currentTree.entries); };
    $("flatMode").onclick = () => { fsMode = "flat"; renderFs(currentTree.entries); };
    $("treeEngineTarget").onclick = async () => {
      $("fsTarget").value = "echo://engine/tree";
      await refreshTree();
    };
    $("chatNav").onclick = () => setSideView("context");
    $("memoryNav").onclick = () => setSideView("memory");
    $("locomoNav").onclick = () => { location.href = "/agent/locomo"; };
    $("graphEvalNav").onclick = () => { location.href = "/agent/graph-eval"; };
    $("memoryRefresh").onclick = refreshInspectors;
    $("memoryChip").onclick = () => $("userText").value = "鐠囧嘲鐔€娴?EchoMemory 濡偓缁便垻绮ㄩ弸婊愮礉鐢喗鍨滈幀鑽ょ波瑜版挸澧犳导姘崇樈娑擃厾娈戦崗鎶芥暛鐠佹澘绻傞妴?;
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
    loadAccounts();
    renderAccounts();
    ensureLocomoAccountOnStartup().finally(() => renderAccounts());
    loadSessions();
    if (sessions.length > 0) setActiveSession(sessions[0]);
    else resetConversation();
    refreshInspectors();
  </script>
</body>
</html>"""

LOCOMO_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>LoCoMo 鐠囧嫭绁撮崣?/title>
  <style>
    :root { --ink:#182235; --muted:#66758b; --line:#d7e0ec; --blue:#2357c6; --green:#16734d; --orange:#a65a16; --red:#a23a3a; }
    * { box-sizing: border-box; }
    body { margin:0; color:var(--ink); background:#eef1f6; font-family:"Segoe UI","Microsoft YaHei",Arial,sans-serif; letter-spacing:0; }
    header { display:flex; align-items:center; justify-content:space-between; gap:16px; padding:14px 18px; border-bottom:1px solid var(--line); background:#fff; }
    h1 { margin:0; font-size:20px; }
    h3 { margin:16px 0 8px; }
    button,input { font:inherit; }
    button { min-height:32px; padding:6px 10px; border:1px solid #c6d2e3; border-radius:6px; background:#fff; color:#24344f; cursor:pointer; font-weight:700; }
    button.primary { border-color:var(--blue); background:var(--blue); color:#fff; }
    button:disabled { cursor:wait; opacity:.65; }
    input { width:100%; min-height:34px; padding:6px 9px; border:1px solid #c6d2e3; border-radius:6px; background:#fff; }
    label { display:grid; gap:5px; color:var(--muted); font-size:12px; font-weight:700; }
    main { display:grid; grid-template-columns:300px minmax(520px,1fr) 420px; gap:12px; padding:12px; min-height:calc(100vh - 61px); }
    section { min-width:0; border:1px solid var(--line); border-radius:8px; background:#fff; overflow:hidden; }
    .head { display:flex; align-items:center; justify-content:space-between; gap:10px; padding:12px; border-bottom:1px solid var(--line); background:#fbfcff; }
    .head h2 { margin:0; font-size:16px; }
    .status,.muted { color:var(--muted); font-size:12px; }
    .toolbar { display:flex; flex-wrap:wrap; gap:8px; padding:10px 12px; border-bottom:1px solid var(--line); background:#f7f9fd; }
    .actionbar { display:none; flex-wrap:wrap; gap:8px; align-items:end; width:100%; }
    .actionbar.active { display:flex; }
    .body { padding:12px; }
    .stack { display:grid; gap:10px; }
    .sample,.session,.qa,.run,.metric,.event { padding:10px; border:1px solid var(--line); border-radius:8px; background:#fff; }
    .sample.active,.session.active,.qa.active { border-color:var(--blue); background:#edf4ff; }
    .session.imported { border-color:#cfd8e5; background:#f7f9fc; color:#536176; cursor:not-allowed; }
    .row { display:flex; align-items:center; justify-content:space-between; gap:8px; font-weight:800; }
    .tag { display:inline-flex; align-items:center; padding:2px 7px; border-radius:999px; background:#eef2f7; color:#43536d; font-size:12px; font-weight:800; }
    .tag.good { background:#edf8f1; color:var(--green); }
    .tag.warn { background:#fff4e4; color:var(--orange); }
    .tag.bad { background:#fff0f0; color:var(--red); }
    .grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; }
    .metric strong { display:block; font-size:22px; line-height:1.15; }
    .tabs { display:flex; border:1px solid #c6d2e3; border-radius:7px; overflow:hidden; }
    .tabs button { border:0; border-right:1px solid #c6d2e3; border-radius:0; }
    .tabs button:last-child { border-right:0; }
    .tabs button.active { background:var(--blue); color:#fff; }
    .scroll { max-height:520px; overflow:auto; }
    .turn { max-width:88%; margin:8px 0; padding:9px 10px; border:1px solid var(--line); border-radius:8px; background:#fff; font-size:13px; line-height:1.45; }
    .turn.alt { margin-left:auto; border-color:#c7d8ff; background:#edf4ff; }
    .answer { border-color:#cce9d9; background:#edf8f1; }
    pre { white-space:pre-wrap; margin:0; font-family:"Cascadia Code",Consolas,monospace; font-size:12px; line-height:1.5; }
    .error { color:var(--red); }
    .hidden { display:none; }
    @media (max-width:1120px) { main { grid-template-columns:1fr; } .grid { grid-template-columns:repeat(2,minmax(0,1fr)); } }
  </style>
</head>
<body>
  <header>
    <div><h1>LoCoMo 鐠囧嫭绁撮崣?/h1><div class="status" id="datasetStatus">閸旂姾娴囬弫鐗堝祦闂嗗棔鑵?..</div></div>
    <div style="display:flex;gap:8px;align-items:center;"><button id="sync">閸掗攱鏌婇弫鐗堝祦闂?/button><button onclick="location.href='/'">鏉╂柨娲栫€电鐦?/button></div>
  </header>
  <main>
    <section>
      <div class="head"><h2>閺嶉攱婀?/h2><span class="tag" id="sampleCount">0</span></div>
      <div class="toolbar"><input id="sampleSearch" placeholder="閹兼粎鍌?sample / speaker" /></div>
      <div class="body scroll"><div class="stack" id="samples"></div></div>
    </section>
    <section>
      <div class="head">
        <div><h2 id="detailTitle">閺嶉攱婀扮拠锔藉剰</h2><div class="status" id="detailSubtitle">闁瀚ㄥ锔挎櫠閺嶉攱婀伴弻銉ф箙鐠囷附鍎?/div></div>
        <div class="tabs"><button class="active" data-tab="import">鐎电厧鍙?/button><button data-tab="evaluate">鐠囧嫭绁?/button><button data-tab="run">鏉╂劘顢?/button><button data-tab="results">缂佹挻鐏?/button></div>
      </div>
      <div class="toolbar">
        <div class="actionbar active" id="actions-import">
          <button id="selectAllSessions">闁瀚ㄩ崗銊╁劥娴兼俺鐦?/button><button id="clearSessions">濞撳懐鈹栨导姘崇樈</button><button class="primary" id="importSessions">鐎电厧鍙嗛幍鈧柅澶夌窗鐠?/button>
          <label style="width:150px;">閻劍鍩?ID<input id="userId" value="locomo-user" /></label>
          <label style="width:160px;">Agent ID<input id="agentId" value="locomo-agent" /></label>
        </div>
        <div class="actionbar" id="actions-evaluate">
          <button id="selectAll">闁瀚ㄩ崗銊╁劥 QA</button><button id="clearSelection">濞撳懐鈹?QA</button><button class="primary" id="startRun">閸欐垼鎹ｇ拠鍕ゴ</button>
          <span class="muted" id="evaluateHint">鐎电厧鍙嗛懛鍐茬毌娑撯偓娑擃亙绱扮拠婵嗘倵閸欘垵鐦庡ù瀣ㄢ偓?/span>
        </div>
        <div class="actionbar" id="actions-run">
          <button id="refreshRunView">閸掗攱鏌婅ぐ鎾冲鏉╂劘顢?/button><button id="openResults">閺屻儳婀呯紒鎾寸亯</button>
        </div>
        <div class="actionbar" id="actions-results">
          <button id="refreshResultsView">閸掗攱鏌婄紒鎾寸亯</button><button id="backToEvaluate">鏉╂柨娲栫拠鍕ゴ</button>
        </div>
      </div>
      <div class="body">
        <div class="tab-pane" id="tab-import">
          <div class="grid">
            <div class="metric"><span class="muted">Sessions</span><strong id="mSessions">-</strong></div>
            <div class="metric"><span class="muted">Imported</span><strong id="mImported">0</strong></div>
            <div class="metric"><span class="muted">Selected Conv</span><strong id="mSelectedSessions">0</strong></div>
            <div class="metric"><span class="muted">Turns</span><strong id="mTurns">-</strong></div>
          </div>
          <h3>娴兼俺鐦界€电厧鍙?/h3><div class="stack scroll" id="sessionList"></div>
          <h3>鐎电鐦芥０鍕潔</h3><div class="scroll" id="turns"></div>
        </div>
        <div class="tab-pane hidden" id="tab-evaluate">
          <div class="grid">
            <div class="metric"><span class="muted">Turns</span><strong id="mEvalTurns">-</strong></div>
            <div class="metric"><span class="muted">QA</span><strong id="mQa">-</strong></div>
            <div class="metric"><span class="muted">Selected</span><strong id="mSelected">0</strong></div>
            <div class="metric"><span class="muted">Ready</span><strong id="mReady">-</strong></div>
          </div>
          <h3>QA</h3><div class="stack scroll" id="qaList"></div>
        </div>
        <div class="tab-pane hidden" id="tab-run"><div class="stack" id="currentRun"></div><h3>娴滃娆?/h3><div class="stack scroll" id="events"></div></div>
        <div class="tab-pane hidden" id="tab-results"><div class="stack scroll" id="results"></div></div>
      </div>
    </section>
    <section>
      <div class="head"><h2>鏉╂劘顢戞稉搴″坊閸?/h2><button id="refreshRuns">閸掗攱鏌?/button></div>
      <div class="body scroll">
        <div class="stack">
          <div class="run">
            <div class="row"><span>鐎圭偞妞傛潻娑樺</span><span class="tag warn" id="sideStatus">idle</span></div>
            <div id="sideProgress" class="muted">闁瀚?QA 閸氬骸褰傜挧鐤槑濞村鈧?/div>
          </div>
          <div>
            <h3>閸ョ偞鏂佺€电鐦?/h3>
            <div class="scroll" id="liveReplay"></div>
          </div>
          <div>
            <h3>闁劙顣界紒鎾寸亯</h3>
            <div class="stack" id="liveScores"></div>
          </div>
          <div>
            <h3>娴滃娆㈠ù?/h3>
            <div class="stack" id="sideEvents"></div>
          </div>
          <div>
            <h3>閸樺棗褰剁拠鍕ゴ</h3>
            <div class="stack" id="runs"></div>
          </div>
        </div>
      </div>
    </section>
  </main>
  <script>
    const $ = (id) => document.getElementById(id);
    let dataset = {samples: []}, activeSample = null, selectedQa = new Set(), selectedSessions = new Set(), importRecords = {}, activeRunId = null, pollTimer = null, locomoAccount = null;
    function escapeHtml(value) { return String(value ?? "").replace(/[&<>"']/g, (char) => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[char])); }
    async function api(path, options = {}) {
      const headers = {"Content-Type": "application/json", ...(options.headers || {})};
      if (locomoAccount?.authKey) headers["X-Auth-Key"] = locomoAccount.authKey;
      const response = await fetch(path, {...options, headers});
      const data = await response.json();
      if (!response.ok || data.error) throw new Error(data.message || data.error || response.statusText);
      return data;
    }
    async function ensureLocomoAccount() {
      const data = await api("/agent/accounts/ensure", {method: "POST", body: JSON.stringify({label: "locomo"})});
      locomoAccount = data.account;
      $("datasetStatus").textContent = `locomo 鐠愶附鍩涘鎻掓皑缂?璺?${locomoAccount?.tenantId || locomoAccount?.id || ""}`;
    }
    async function loadDataset() {
      dataset = await api("/agent/locomo/dataset");
      await loadImports();
      const manifest = dataset.manifest || {};
      $("datasetStatus").textContent = `${manifest.exists ? "閺堫剙婀撮弫鐗堝祦闂嗗棗鍑＄亸杈╁崕" : "閺佺増宓侀梿鍡樻弓娑撳娴?} 璺?${manifest.path || ""}`;
      $("sampleCount").textContent = `${dataset.samples.length} samples`;
      renderSamples();
    }
    async function loadImports() {
      const data = await api("/agent/locomo/imports");
      importRecords = data.by_sample || {};
    }
    function renderSamples() {
      const q = $("sampleSearch").value.trim().toLowerCase();
      const items = dataset.samples.filter((item) => !q || JSON.stringify(item).toLowerCase().includes(q));
      $("samples").innerHTML = items.map((item) => `
        <div class="sample ${activeSample?.sample_id === item.sample_id ? "active" : ""}" data-sample="${escapeHtml(item.sample_id)}">
          <div class="row"><span>${escapeHtml(item.sample_id)}</span><span class="tag ${importedCount(item.sample_id) ? "good" : ""}">${importedCount(item.sample_id)}/${item.session_count} imported</span></div>
          <div class="muted">${escapeHtml(item.speaker_a || "-")} / ${escapeHtml(item.speaker_b || "-")} 璺?${item.session_count} sessions 璺?${item.turn_count} turns</div>
          <div class="muted">${item.qa_count} QA</div>
        </div>`).join("");
      [...document.querySelectorAll(".sample")].forEach((node) => node.onclick = () => loadSample(node.dataset.sample));
    }
    function importedCount(sampleId) {
      return (importRecords[sampleId]?.sessions || []).length;
    }
    async function loadSample(sampleId) {
      activeSample = await api(`/agent/locomo/dataset/${encodeURIComponent(sampleId)}`);
      selectedQa = new Set();
      selectedSessions = new Set((activeSample.sessions || []).filter((session) => !isSessionImported(session.id)).map((session) => session.id));
      $("detailTitle").textContent = `${activeSample.sample_id} 鐠囷附鍎廯;
      $("detailSubtitle").textContent = `${activeSample.speaker_a || "-"} / ${activeSample.speaker_b || "-"}`;
      renderSamples(); renderSample();
    }
    function importRecord() {
      return activeSample ? (importRecords[activeSample.sample_id] || {sessions: []}) : {sessions: []};
    }
    function isSessionImported(sessionId) {
      return new Set(importRecord().sessions || []).has(sessionId);
    }
    function renderSample() {
      const sessions = activeSample?.sessions || [], qa = activeSample?.qa || [];
      const imported = new Set(importRecord().sessions || []);
      $("mSessions").textContent = sessions.length;
      $("mTurns").textContent = sessions.reduce((sum, item) => sum + item.turn_count, 0);
      $("mImported").textContent = imported.size;
      $("mSelectedSessions").textContent = selectedSessions.size;
      $("mEvalTurns").textContent = sessions.reduce((sum, item) => sum + item.turn_count, 0);
      $("mQa").textContent = qa.length; $("mSelected").textContent = selectedQa.size;
      $("mReady").textContent = imported.size > 0 ? "yes" : "no";
      $("evaluateHint").textContent = imported.size > 0
        ? `瀹告彃顕遍崗?${imported.size} 娑擃亙绱扮拠婵撶礉閸欘垶鈧瀚?QA 閸欐垼鎹ｇ拠鍕ゴ閵嗕繖
        : "鐎电厧鍙嗛懛鍐茬毌娑撯偓娑擃亙绱扮拠婵嗘倵閸欘垵鐦庡ù瀣ㄢ偓?;
      $("sessionList").innerHTML = sessions.map((session) => {
        const done = imported.has(session.id);
        const checked = selectedSessions.has(session.id);
        return `
          <div class="session ${checked ? "active" : ""} ${done ? "imported" : ""}" data-session="${escapeHtml(session.id)}">
            <div class="row"><span>${escapeHtml(session.id)}</span><span class="tag ${done ? "good" : ""}">${done ? "瀹告彃顕遍崗? : `${session.turn_count} turns`}</span></div>
            <div class="muted">${escapeHtml(session.date_time || "")}</div>
            <div class="muted">${escapeHtml(session.summary || session.observation || "")}</div>
          </div>`;
      }).join("") || "<div class='muted'>閺嗗倹妫ゆ导姘崇樈</div>";
      [...document.querySelectorAll(".session")].forEach((node) => node.onclick = (event) => {
        if (isSessionImported(node.dataset.session)) return;
        if (selectedSessions.has(node.dataset.session)) selectedSessions.delete(node.dataset.session);
        else selectedSessions.add(node.dataset.session);
        renderSample();
      });
      $("qaList").innerHTML = qa.map((item) => `
        <div class="qa ${selectedQa.has(item.id) ? "active" : ""}" data-qa="${escapeHtml(item.id)}">
          <div class="row"><span>#${item.index} ${escapeHtml(item.category)}</span><span class="tag">${escapeHtml(item.id)}</span></div>
          <div>${escapeHtml(item.question)}</div><div class="muted">Answer: ${escapeHtml(item.answer)}</div>
          <div class="muted">Evidence: ${escapeHtml((item.evidence || []).join(", "))}</div>
        </div>`).join("");
      [...document.querySelectorAll(".qa")].forEach((node) => node.onclick = () => { selectedQa.has(node.dataset.qa) ? selectedQa.delete(node.dataset.qa) : selectedQa.add(node.dataset.qa); renderSample(); });
      const turns = sessions.slice(0, 2).flatMap((session) => (session.turns || []).slice(0, 10).map((turn, index) => ({...turn, session: session.id, index})));
      $("turns").innerHTML = turns.map((turn) => `<div class="turn ${turn.index % 2 ? "alt" : ""}"><b>${escapeHtml(turn.speaker)} 璺?${escapeHtml(turn.session)} 璺?${escapeHtml(turn.dia_id || "")}</b><br>${escapeHtml(turn.text || "")}</div>`).join("") || "<div class='muted'>閺嗗倹妫ょ€电鐦?/div>";
    }
    function setTab(tab) {
      [...document.querySelectorAll(".tabs button")].forEach((button) => button.classList.toggle("active", button.dataset.tab === tab));
      [...document.querySelectorAll(".tab-pane")].forEach((pane) => pane.classList.add("hidden"));
      $(`tab-${tab}`).classList.remove("hidden");
      [...document.querySelectorAll(".actionbar")].forEach((bar) => bar.classList.remove("active"));
      const actions = $(`actions-${tab}`);
      if (actions) actions.classList.add("active");
    }
    async function startRun() {
      if (!activeSample) return;
      if ((importRecord().sessions || []).length === 0) throw new Error("鐠囧嘲鍘涚€电厧鍙嗛懛鍐茬毌娑撯偓娑擃亙绱扮拠?);
      const qaIds = selectedQa.size ? [...selectedQa] : activeSample.qa.map((item) => item.id);
      $("startRun").disabled = true;
      try {
        const run = await api("/agent/locomo/runs", {method: "POST", body: JSON.stringify({sample_ids: [activeSample.sample_id], qa_ids: {[activeSample.sample_id]: qaIds}, user_id: $("userId").value.trim(), agent_id: $("agentId").value.trim()})});
        activeRunId = run.run_id; setTab("run"); pollRun();
      } finally { $("startRun").disabled = false; }
    }
    async function importSelectedSessions() {
      if (!activeSample) return;
      const sessionIds = [...selectedSessions].filter((id) => !isSessionImported(id));
      if (sessionIds.length === 0) return;
      $("importSessions").disabled = true;
      try {
        const data = await api("/agent/locomo/imports", {
          method: "POST",
          body: JSON.stringify({
            sample_ids: [activeSample.sample_id],
            session_ids: {[activeSample.sample_id]: sessionIds},
            user_id: $("userId").value.trim(),
            agent_id: $("agentId").value.trim()
          })
        });
        await loadImports();
        selectedSessions = new Set((activeSample.sessions || []).filter((session) => !isSessionImported(session.id)).map((session) => session.id));
        renderSample();
        $("sideStatus").textContent = "imported";
        $("sideProgress").textContent = JSON.stringify(data.imports?.[0] || data, null, 2);
      } finally {
        $("importSessions").disabled = false;
      }
    }
    async function pollRun() {
      if (!activeRunId) return;
      const [run, events] = await Promise.all([api(`/agent/locomo/runs/${encodeURIComponent(activeRunId)}`), api(`/agent/locomo/runs/${encodeURIComponent(activeRunId)}/events`)]);
      renderRun(run, events.events || []); await loadRuns();
      if (["queued", "running"].includes(run.status)) { clearTimeout(pollTimer); pollTimer = setTimeout(pollRun, 1200); }
    }
    function renderRun(run, events) {
      const summary = run.summary || {}, progress = run.progress || {};
      $("currentRun").innerHTML = `<div class="run"><div class="row"><span>${escapeHtml(run.run_id)}</span><span class="tag ${run.status === "completed" ? "good" : run.status === "failed" ? "bad" : "warn"}">${escapeHtml(run.status)}</span></div><div class="muted">鐎瑰本鍨?${progress.completed || 0} / ${progress.total || 0} 璺?F1 ${summary.token_f1 ?? "-"} 璺?contains ${summary.contains ?? "-"}</div>${run.error ? `<div class="error">${escapeHtml(run.error)}</div>` : ""}</div>`;
      $("events").innerHTML = events.slice(-80).reverse().map((event) => `<div class="event"><pre>${escapeHtml(JSON.stringify(event, null, 2))}</pre></div>`).join("");
      $("results").innerHTML = (run.results || []).map((item) => `<div class="run"><div class="row"><span>${escapeHtml(item.sample_id)} 璺?${escapeHtml(item.qa_id)}</span><span class="tag good">F1 ${item.scores?.token_f1 ?? 0}</span></div><div><b>Q:</b> ${escapeHtml(item.question)}</div><div><b>Gold:</b> ${escapeHtml(item.gold_answer)}</div><div class="answer turn"><b>Agent:</b><br>${escapeHtml(item.agent_answer)}</div>${item.error ? `<div class="error">${escapeHtml(item.error)}</div>` : ""}</div>`).join("") || "<div class='muted'>閺嗗倹妫ょ紒鎾寸亯</div>";
      $("sideStatus").textContent = run.status || "idle";
      $("sideStatus").className = `tag ${run.status === "completed" ? "good" : run.status === "failed" ? "bad" : "warn"}`;
      $("sideProgress").innerHTML = `run: ${escapeHtml(run.run_id)}<br>鐎瑰本鍨?${progress.completed || 0} / ${progress.total || 0}<br>token_f1 ${summary.token_f1 ?? "-"} 璺?contains ${summary.contains ?? "-"} 璺?errors ${summary.errors ?? 0}`;
      const replay = events.filter((event) => event.type === "turn_replayed").slice(-80);
      $("liveReplay").innerHTML = replay.map((event, index) => `<div class="turn ${index % 2 ? "alt" : ""}"><b>${escapeHtml(event.speaker)} 璺?${escapeHtml(event.session)} 璺?${escapeHtml(event.dia_id || "")}</b><br>${escapeHtml(event.text || "")}</div>`).join("") || "<div class='muted'>缁涘绶熼崶鐐存杹鐎电鐦?..</div>";
      $("liveScores").innerHTML = (run.results || []).map((item) => `<div class="run"><div class="row"><span>${escapeHtml(item.qa_id)}</span><span class="tag ${item.error ? "bad" : "good"}">F1 ${item.scores?.token_f1 ?? 0}</span></div><div class="muted">${escapeHtml(item.category)} 璺?contains ${item.scores?.contains ?? 0}</div><div>${escapeHtml(item.question)}</div></div>`).join("") || "<div class='muted'>缁涘绶熺拠鍕瀻...</div>";
      $("sideEvents").innerHTML = events.slice(-12).reverse().map((event) => `<div class="event"><div class="row"><span>${escapeHtml(event.type)}</span><span class="muted">${escapeHtml(event.ts || "")}</span></div></div>`).join("") || "<div class='muted'>閺嗗倹妫ゆ禍瀣╂</div>";
    }
    async function loadRuns() {
      const data = await api("/agent/locomo/runs");
      $("runs").innerHTML = (data.runs || []).map((run) => `<div class="run" data-run="${escapeHtml(run.run_id)}"><div class="row"><span>${escapeHtml(run.run_id)}</span><span class="tag ${run.status === "completed" ? "good" : run.status === "failed" ? "bad" : "warn"}">${escapeHtml(run.status)}</span></div><div class="muted">F1 ${run.summary?.token_f1 ?? "-"} 璺?${run.progress?.completed ?? 0}/${run.progress?.total ?? 0} 璺?${escapeHtml(run.created_at || "")}</div></div>`).join("") || "<div class='muted'>閺嗗倹妫ら崢鍡楀蕉鐠囧嫭绁?/div>";
      [...document.querySelectorAll(".run[data-run]")].forEach((node) => node.onclick = () => { activeRunId = node.dataset.run; setTab("run"); pollRun(); });
    }
    $("sync").onclick = async () => { await api("/agent/locomo/dataset/sync", {method:"POST", body:JSON.stringify({force:true})}); await loadDataset(); };
    $("sampleSearch").oninput = renderSamples;
    $("selectAllSessions").onclick = () => { if (activeSample) { selectedSessions = new Set(activeSample.sessions.filter((item) => !isSessionImported(item.id)).map((item) => item.id)); renderSample(); } };
    $("clearSessions").onclick = () => { selectedSessions = new Set(); renderSample(); };
    $("importSessions").onclick = () => importSelectedSessions().catch((error) => $("sideProgress").innerHTML = `<span class="error">${escapeHtml(error.message)}</span>`);
    $("selectAll").onclick = () => { if (activeSample) { selectedQa = new Set(activeSample.qa.map((item) => item.id)); renderSample(); } };
    $("clearSelection").onclick = () => { selectedQa = new Set(); renderSample(); };
    $("startRun").onclick = () => startRun().catch((error) => $("sideProgress").innerHTML = `<span class="error">${escapeHtml(error.message)}</span>`);
    $("refreshRuns").onclick = loadRuns;
    $("refreshRunView").onclick = () => activeRunId ? pollRun() : loadRuns();
    $("openResults").onclick = () => setTab("results");
    $("refreshResultsView").onclick = () => activeRunId ? pollRun() : loadRuns();
    $("backToEvaluate").onclick = () => setTab("evaluate");
    [...document.querySelectorAll(".tabs button")].forEach((button) => button.onclick = () => setTab(button.dataset.tab));
    ensureLocomoAccount().then(loadDataset).catch((error) => $("datasetStatus").innerHTML = `<span class="error">${escapeHtml(error.message)}</span>`);
    loadRuns().catch(() => {});
  </script>
</body>
</html>"""


GRAPH_EVAL_HTML = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8" /><meta name="viewport" content="width=device-width, initial-scale=1" /><title>Graph Eval</title>
<style>body{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#f6f8fc;color:#162033}header{padding:12px 16px;background:#fff;border-bottom:1px solid #e5eaf3;display:flex;justify-content:space-between;align-items:center}main{display:grid;grid-template-columns:280px 1fr;gap:12px;padding:12px}aside,.panel{background:#fff;border:1px solid #e5eaf3;border-radius:12px;padding:12px}.tabs{display:flex;gap:8px;margin-bottom:10px}.tab-button{border:1px solid #d8e1f0;background:#fff;border-radius:999px;padding:6px 10px;cursor:pointer}.tab-button.active{background:#2363eb;color:#fff;border-color:#2363eb}.hidden{display:none}.case{border:1px solid #e5eaf3;border-radius:10px;padding:8px;margin-bottom:8px;cursor:pointer}.case.active{border-color:#2363eb;background:#f0f5ff}.muted{color:#6b7385;font-size:12px}.workspace{display:grid;grid-template-columns:1.1fr 0.9fr;gap:10px}.panel-title{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;gap:8px}.dialogue-list{max-height:320px;overflow:auto;display:grid;gap:8px}.turn{border:1px solid #ebeff6;border-radius:10px;padding:8px}.turn-head{display:flex;justify-content:space-between;margin-bottom:4px}.qa-list{display:grid;gap:8px;max-height:320px;overflow:auto}.qa{border:1px solid #e5eaf3;border-radius:10px;padding:8px;display:flex;gap:8px;align-items:flex-start}.result{border:1px solid #e5eaf3;border-radius:10px;padding:8px;margin-bottom:8px}.pass{color:#0f8a39;font-weight:700}.fail{color:#c43a2e;font-weight:700}.pill{border-radius:999px;padding:2px 8px;border:1px solid #d5deed;font-size:12px}.import-row{border:1px solid #e5eaf3;border-radius:10px;padding:8px;margin-bottom:8px;cursor:pointer}.import-row.active{border-color:#2363eb;background:#f0f5ff}button{border:1px solid #d5deed;background:#fff;border-radius:8px;padding:6px 10px;cursor:pointer}button.primary{background:#2363eb;color:#fff;border-color:#2363eb}pre{white-space:pre-wrap;word-break:break-word;background:#f7f9fe;border:1px solid #e5eaf3;border-radius:8px;padding:8px}</style></head>
<body><header><div><strong>Graph Eval</strong><div class="muted" id="status">ready</div></div><div><span class="muted">当前账户</span><strong id="currentAccountLabel">local account</strong><select id="accountSelect"></select><button type="button" id="createAccount">新建</button><button type="button" id="forgetAccount">移除</button><button type="button" onclick="location.href='/'">返回对话</button></div></header>
<main><aside><div class="panel-title"><strong>评测集</strong><span class="muted" id="caseCount">0</span></div><div id="cases"></div></aside><section><div class="tabs"><button class="tab-button active" id="importTab">导入数据</button><button class="tab-button" id="evalTab">执行评测</button></div><div class="tab-panel" id="importPanel"><div class="workspace"><div><div class="panel"><div class="panel-title"><div><strong id="caseTitle">Graph Eval</strong><div class="muted" id="caseDesc"></div></div><button id="importHistory">导入历史对话</button></div><div class="muted" id="importState">尚未导入当前评测集</div></div><div class="panel"><div class="panel-title"><strong>历史对话预览</strong><span class="muted" id="dialogueStatus">0 turns</span></div><div class="dialogue-list" id="dialogueList"></div></div></div><div class="panel"><div class="panel-title"><strong>导入记录</strong><button id="refreshImports">刷新</button></div><div id="imports"></div></div></div></div><div class="tab-panel hidden" id="evalPanel"><div class="workspace"><div><div class="panel"><div class="panel-title"><div><strong>评测控制</strong><div class="muted" id="evalState">请先导入历史对话，再执行评测</div></div><button class="primary" id="run">运行评测</button></div></div><div class="panel"><div class="panel-title"><strong>问题选择</strong><div><button id="selectAllQa">全选</button><button id="clearQa">清空</button></div></div><div class="qa-list" id="qaList"></div></div><div class="panel"><div class="panel-title"><strong>评测结果</strong><button id="refreshRuns">刷新</button></div><div id="results"></div></div></div><div class="panel"><div class="panel-title"><strong>原始响应</strong></div><pre id="raw">{}</pre></div></div></div></section></main>
<script>
const accountStoreKey="echomem-agent.accounts.v1";let accounts=[],activeAccount=null,cases=[],selected=null,selectedQa=new Set(),activeImport=null,visibleImports=[];const $=(id)=>document.getElementById(id);
async function api(path,options={}){const headers=Object.assign({},options.headers||{});if(activeAccount?.authKey)headers["X-Auth-Key"]=activeAccount.authKey;const response=await fetch(path,{...options,headers});const data=await response.json();if(!response.ok||data.error)throw new Error(data.message||data.error||response.statusText);return data;}
function escapeHtml(value){return String(value??"").replace(/[&<>"']/g,(ch)=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));}
function loadAccounts(){try{const parsed=JSON.parse(localStorage.getItem(accountStoreKey)||"[]");accounts=Array.isArray(parsed)?parsed:[]}catch(_){accounts=[]}if(!accounts.some((item)=>item.id==="local"))accounts.unshift({id:"local",label:"local account",authKey:"",tenantId:"local",userId:"local"});const activeId=localStorage.getItem(`${accountStoreKey}.active`)||"";activeAccount=accounts.find((item)=>item.id===activeId)||accounts[0];saveAccounts();}
function saveAccounts(){localStorage.setItem(accountStoreKey,JSON.stringify(accounts.slice(0,20)));localStorage.setItem(`${accountStoreKey}.active`,activeAccount?.id||"local");renderAccounts();}
function renderAccounts(){$("accountSelect").innerHTML=accounts.map((item)=>`<option value="${escapeHtml(item.id)}" ${item.id===activeAccount?.id?"selected":""}>${escapeHtml(item.label||item.id)}</option>`).join("");$("currentAccountLabel").textContent=activeAccount?.label||"local account";$("forgetAccount").disabled=!activeAccount||activeAccount.id==="local";}
function accountPayload(){return{account_id:activeAccount?.id||"local",tenant_id:activeAccount?.tenantId||activeAccount?.id||"local",user_id:activeAccount?.userId||activeAccount?.id||"local"};}
function switchTab(tab){$("importPanel").classList.toggle("hidden",tab!=="import");$("evalPanel").classList.toggle("hidden",tab!=="eval");$("importTab").classList.toggle("active",tab==="import");$("evalTab").classList.toggle("active",tab==="eval");updateImportState();}
async function createAccount(){const label=`graph eval ${accounts.filter((item)=>item.id!=="local").length+1}`;$("status").textContent=`creating account ${label}...`;const data=await api("/agent/accounts/create",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({label})});const account=data.account;if(!account?.authKey)throw new Error("account creation returned no authKey");activeAccount={id:account.id,label:account.label,tenantId:account.tenantId,userId:account.userId,authKey:account.authKey};accounts=[activeAccount,...accounts.filter((item)=>item.id!==activeAccount.id)];saveAccounts();activeImport=null;await loadImports();$("status").textContent=`switched to ${activeAccount.label}`;}
async function forgetAccount(){if(!activeAccount||activeAccount.id==="local")return;const removing=activeAccount;await api("/agent/accounts/forget",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({id:removing.id,label:removing.label})});accounts=accounts.filter((item)=>item.id!==removing.id);activeAccount=accounts[0]||{id:"local",label:"local account",authKey:"",tenantId:"local",userId:"local"};saveAccounts();activeImport=null;renderImports([]);updateImportState();}
function caseById(id){return cases.find((item)=>item.id===id);}
async function loadCases(){const data=await api("/agent/graph-eval/cases");cases=data.cases||[];selected=selected||cases[0]?.id||null;selectedQa=new Set((caseById(selected)?.queries||[]).slice(0,3).map((item)=>item.id));$("caseCount").textContent=`${cases.length} cases`;renderCases();await Promise.all([loadImports(),loadRuns()]);}
function renderCases(){$("cases").innerHTML=cases.map((item)=>`<div class="case ${item.id===selected?"active":""}" data-id="${escapeHtml(item.id)}"><strong>${escapeHtml(item.title)}</strong><div class="muted">${item.ingest_count} turns / ${item.query_count} QA</div></div>`).join("");[...document.querySelectorAll(".case[data-id]")].forEach((node)=>node.onclick=async()=>{selected=node.dataset.id;selectedQa=new Set((caseById(selected)?.queries||[]).slice(0,3).map((item)=>item.id));activeImport=null;renderCases();await loadImports();});const item=caseById(selected)||{};$("caseTitle").textContent=item.title||"Graph Eval";$("caseDesc").textContent=item.description||"";renderDialogue();renderQa();updateImportState();}
function renderDialogue(){const turns=caseById(selected)?.dialogue||[];$("dialogueStatus").textContent=`${turns.length} turns`;$("dialogueList").innerHTML=turns.map((turn)=>`<div class="turn"><div class="turn-head"><span>${escapeHtml(turn.role)}</span><span class="muted">${escapeHtml(turn.date||"")} / ${escapeHtml(turn.timestamp||"")}</span></div><div>${escapeHtml(turn.content||"")}</div></div>`).join("");}
function renderQa(){const queries=caseById(selected)?.queries||[];$("qaList").innerHTML=queries.map((item)=>`<label class="qa"><input type="checkbox" value="${escapeHtml(item.id)}" ${selectedQa.has(item.id)?"checked":""}/><span><strong>#${escapeHtml(item.number||"?")} ${escapeHtml(item.name||"")}</strong><div class="muted">${escapeHtml(item.question_id||item.id||"")}</div><div class="muted">${escapeHtml(item.query||"")}</div>${item.expected?`<div class="muted">Expected: ${escapeHtml(item.expected)}</div>`:""}</span></label>`).join("");[...document.querySelectorAll("#qaList input")].forEach((box)=>box.onchange=()=>{if(box.checked)selectedQa.add(box.value);else selectedQa.delete(box.value);});}
function updateImportState(){const imported=activeImport?.status==="imported"&&activeImport?.case_id===selected;$("importHistory").disabled=Boolean(imported);$("run").disabled=!imported;if(imported){$("importState").textContent=`已导入 ${activeImport.imported_count}/${activeImport.total_count} turns，记录 ${activeImport.import_id}`;$("evalState").textContent="可选择一个或多个问题执行评测";}else if(activeImport?.status==="running"){$("importState").textContent=`导入中 ${activeImport.imported_count||0}/${activeImport.total_count||0}...`;$("evalState").textContent="导入进行中，完成后才能评测";}else if(activeImport?.status==="failed"){$("importState").textContent=`导入失败: ${activeImport.error||"unknown error"}`;$("evalState").textContent="导入失败，请重试";}else{$("importState").textContent="尚未导入当前评测集";$("evalState").textContent="请先导入历史对话";}}
async function importHistory(){$("status").textContent="importing history...";$("importHistory").disabled=true;const data=await api("/agent/graph-eval/imports",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({case_id:selected,...accountPayload()})});activeImport=data;$("raw").textContent=JSON.stringify(data,null,2);await loadImports();$("status").textContent=data.already_imported?"already imported":"import done";updateImportState();if(activeImport?.status==="imported")switchTab("eval");}
async function runEval(){if(selectedQa.size===0)throw new Error("请至少选择一个问题");if(!activeImport||activeImport.status!=="imported")throw new Error("请先导入历史对话");$("status").textContent="running eval...";const data=await api("/agent/graph-eval/runs",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({case_id:selected,import_id:activeImport.import_id,qa_ids:[...selectedQa],...accountPayload()})});renderEvalRun(data);$("raw").textContent=JSON.stringify(data,null,2);$("status").textContent=data.status||"done";await loadRuns();}
async function loadImports(){const data=await api("/agent/graph-eval/imports");const scope=accountPayload();visibleImports=(data.imports||[]).filter((item)=>item.tenant_id===scope.tenant_id);const current=visibleImports.find((item)=>item.case_id===selected&&item.status==="imported")||visibleImports.find((item)=>item.case_id===selected)||null;activeImport=current;renderImports(visibleImports);updateImportState();}
function renderImports(rows){$("imports").innerHTML=rows.length?rows.map((item)=>`<div class="import-row ${item.import_id===activeImport?.import_id?"active":""}" data-id="${escapeHtml(item.import_id)}"><strong>${escapeHtml(item.import_id)}</strong><div><span class="pill ${item.status==="imported"?"pass":item.status==="failed"?"fail":""}">${escapeHtml(item.status||"unknown")}</span></div><div class="muted">${escapeHtml(item.session_id||"")} / ${item.imported_count||0}/${item.total_count||0}</div></div>`).join(""):`<div class="muted">暂无导入记录</div>`;[...document.querySelectorAll(".import-row[data-id]")].forEach((node)=>node.onclick=()=>{activeImport=rows.find((item)=>item.import_id===node.dataset.id)||null;renderImports(rows);updateImportState();});}
async function loadRuns(){const data=await api("/agent/graph-eval/runs");const scope=accountPayload();const run=(data.runs||[]).find((item)=>item.tenant_id===scope.tenant_id&&item.case_id===selected)||null;if(run)renderEvalRun(run);}
function renderEvalRun(run){const summary=run.summary||{};$("results").innerHTML=`<div class="result"><strong>${escapeHtml(run.run_id)}</strong> <span class="${run.status==="passed"?"pass":"fail"}">${escapeHtml(run.status||"unknown")}</span><div class="muted">${escapeHtml(run.case_title||run.case_id||"")}</div><div class="muted">${escapeHtml(run.tenant_id||"")} / ${escapeHtml(run.session_id||"")}</div><div class="muted">Passed ${summary.passed??"-"} / ${summary.total??"-"}</div></div>`+(run.results||[]).map((item)=>{const memories=(item.items||[]).map((m)=>`<div class="muted">${escapeHtml(m.kind||"memory")}</div><div>${escapeHtml(m.text||m.content||"")}</div>`).join("");const answer=item.answer?`<div class="muted">Answer</div><pre>${escapeHtml(item.answer)}</pre>`:"";const expected=item.expected?`<div class="muted">Expected: ${escapeHtml(item.expected)}</div>`:"";const meta=[item.id||"",item.section||"",item.score!=null?`score=${item.score}`:""] .filter(Boolean).join(" / ");return `<div class="result"><div><strong>#${escapeHtml(item.number||"?")} ${escapeHtml(item.name||"")}</strong> <span class="${item.passed?"pass":"fail"}">${item.passed?"PASS":"FAIL"}</span></div><div class="muted">${escapeHtml(meta)}</div><div class="muted">${escapeHtml((item.failures||[]).join(", ")||"通过")}</div>${expected}<pre>${escapeHtml(item.context||"")}</pre>${answer}${memories}</div>`;}).join("");}
$("accountSelect").onchange=()=>{activeAccount=accounts.find((item)=>item.id===$("accountSelect").value)||accounts[0];saveAccounts();activeImport=null;loadImports().catch((error)=>$("status").textContent=error.message);};$("createAccount").onclick=()=>createAccount().catch((error)=>$("status").textContent=error.message);$("forgetAccount").onclick=()=>forgetAccount().catch((error)=>$("status").textContent=error.message);$("importTab").onclick=()=>switchTab("import");$("evalTab").onclick=()=>switchTab("eval");$("importHistory").onclick=()=>importHistory().catch((error)=>{$("status").textContent=error.message;updateImportState();});$("run").onclick=()=>runEval().catch((error)=>$("status").textContent=error.message);$("selectAllQa").onclick=()=>{selectedQa=new Set((caseById(selected)?.queries||[]).map((item)=>item.id));renderQa();};$("clearQa").onclick=()=>{selectedQa=new Set();renderQa();};$("refreshImports").onclick=()=>loadImports().catch((error)=>$("status").textContent=error.message);$("refreshRuns").onclick=()=>loadRuns().catch((error)=>$("status").textContent=error.message);loadAccounts();loadCases().catch((error)=>$("status").textContent=error.message);
</script></body></html>"""

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
        if path == "/agent/locomo":
            self._send_html(LOCOMO_HTML)
            return
        if path == "/agent/graph-eval":
            self._send_html(GRAPH_EVAL_HTML)
            return
        if path == "/agent/config":
            self._send_json(HTTPStatus.OK, self.server.config.public_dict())
            return
        if path == "/agent/graph-eval/cases":
            self._graph_eval_cases()
            return
        if path == "/agent/graph-eval/runs":
            self._graph_eval_runs()
            return
        if path == "/agent/graph-eval/imports":
            self._graph_eval_imports()
            return
        if path == "/agent/locomo/dataset":
            self._locomo_dataset()
            return
        if path.startswith("/agent/locomo/dataset/"):
            self._locomo_sample(unquote(path.rsplit("/", 1)[1]))
            return
        if path == "/agent/locomo/runs":
            self._locomo_runs()
            return
        if path == "/agent/locomo/imports":
            self._locomo_imports()
            return
        if path.startswith("/agent/locomo/runs/"):
            self._locomo_run_get(path)
            return
        if path.startswith("/agent/inspect/") or path.startswith("/api/"):
            self._proxy()
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/agent/context":
            self._agent_context()
            return
        if path == "/agent/chat":
            self._agent_chat()
            return
        if path == "/agent/session/commit":
            self._agent_commit()
            return
        if path == "/agent/accounts/ensure":
            self._agent_account_ensure()
            return
        if path == "/agent/accounts/create":
            self._agent_account_create()
            return
        if path == "/agent/accounts/forget":
            self._agent_account_forget()
            return
        if path == "/agent/locomo/dataset/sync":
            self._locomo_dataset_sync()
            return
        if path == "/agent/locomo/imports":
            self._locomo_import_create()
            return
        if path == "/agent/locomo/runs":
            self._locomo_run_create()
            return
        if path == "/agent/graph-eval/runs":
            self._graph_eval_run_create()
            return
        if path == "/agent/graph-eval/imports":
            self._graph_eval_import_create()
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
            headers = {"Content-Type": self.headers.get("Content-Type", "application/json")}
            auth_key = self.headers.get("X-Auth-Key", "").strip()
            if auth_key:
                headers["X-Auth-Key"] = auth_key
            request = Request(
                target,
                data=body if self.command in {"POST", "PUT", "PATCH"} else None,
                method=self.command,
                headers=headers,
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
            data = AgentChatService(self._request_config()).chat(payload)
            self._send_json(HTTPStatus.OK, data)
        except ValueError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": str(exc)})
        except EchoMemoryClientError as exc:
            self._send_json(HTTPStatus.BAD_GATEWAY, {"error": "echomemory_error", "message": str(exc)})
        except ModelClientError as exc:
            self._send_json(HTTPStatus.BAD_GATEWAY, {"error": "model_error", "message": str(exc)})

    def _agent_context(self) -> None:
        try:
            payload = self._read_json()
            data = AgentChatService(self._request_config()).preview_context(payload)
            self._send_json(HTTPStatus.OK, data)
        except ValueError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": str(exc)})
        except EchoMemoryClientError as exc:
            self._send_json(HTTPStatus.BAD_GATEWAY, {"error": "echomemory_error", "message": str(exc)})

    def _agent_commit(self) -> None:
        try:
            payload = self._read_json()
            session_id = str(payload.get("session_id") or "").strip()
            if not session_id:
                raise ValueError("session_id is required")
            data = AgentChatService(self._request_config()).memory.commit(session_id)
            self._send_json(HTTPStatus.OK, data)
        except ValueError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "bad_request", "message": str(exc)})
        except EchoMemoryClientError as exc:
            self._send_json(HTTPStatus.BAD_GATEWAY, {"error": "echomemory_error", "message": str(exc)})

    def _locomo_dataset_sync(self) -> None:
        try:
            payload = self._read_json()
            data = LocomoDatasetService(self._request_config().locomo).ensure_dataset(force=bool(payload.get("force")))
            self._send_json(HTTPStatus.OK, data)
        except LocomoDatasetError as exc:
            self._send_json(HTTPStatus.BAD_GATEWAY, {"error": "locomo_dataset_error", "message": str(exc)})

    def _graph_eval_cases(self) -> None:
        self._send_json(HTTPStatus.OK, GraphEvalService(self._request_config()).cases())

    def _graph_eval_runs(self) -> None:
        self._send_json(HTTPStatus.OK, GraphEvalService(self._request_config()).list_runs())

    def _graph_eval_imports(self) -> None:
        self._send_json(HTTPStatus.OK, GraphEvalService(self._request_config()).list_imports())

    def _graph_eval_import_create(self) -> None:
        try:
            self._send_json(HTTPStatus.OK, GraphEvalService(self._request_config()).import_case(self._read_json()))
        except (ValueError, OSError, HTTPError, URLError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "graph_eval_error", "message": str(exc)})

    def _graph_eval_run_create(self) -> None:
        try:
            self._send_json(HTTPStatus.OK, GraphEvalService(self._request_config()).run(self._read_json()))
        except (ValueError, OSError, HTTPError, URLError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "graph_eval_error", "message": str(exc)})

    def _locomo_dataset(self) -> None:
        try:
            data = LocomoDatasetService(self._request_config().locomo).index()
            self._send_json(HTTPStatus.OK, data)
        except LocomoDatasetError as exc:
            self._send_json(HTTPStatus.BAD_GATEWAY, {"error": "locomo_dataset_error", "message": str(exc)})

    def _locomo_sample(self, sample_id: str) -> None:
        try:
            data = LocomoDatasetService(self._request_config().locomo).sample(sample_id)
            self._send_json(HTTPStatus.OK, data)
        except LocomoDatasetError as exc:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "locomo_dataset_error", "message": str(exc)})

    def _locomo_run_create(self) -> None:
        try:
            data = LocomoEvalService(self._request_config()).create_run(self._read_json())
            self._send_json(HTTPStatus.OK, data)
        except (LocomoDatasetError, LocomoEvalError, ValueError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "locomo_eval_error", "message": str(exc)})

    def _locomo_runs(self) -> None:
        try:
            self._send_json(HTTPStatus.OK, LocomoEvalService(self._request_config()).list_runs())
        except LocomoEvalError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "locomo_eval_error", "message": str(exc)})

    def _locomo_imports(self) -> None:
        try:
            self._send_json(HTTPStatus.OK, LocomoEvalService(self._request_config()).import_status())
        except LocomoEvalError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "locomo_eval_error", "message": str(exc)})

    def _locomo_import_create(self) -> None:
        try:
            data = LocomoEvalService(self._request_config()).import_samples(self._read_json())
            self._send_json(HTTPStatus.OK, data)
        except (LocomoDatasetError, LocomoEvalError, ValueError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "locomo_eval_error", "message": str(exc)})

    def _locomo_run_get(self, path: str) -> None:
        try:
            parts = [part for part in path.split("/") if part]
            if len(parts) == 5 and parts[-1] == "events":
                data = LocomoEvalService(self._request_config()).get_events(parts[-2])
            elif len(parts) == 4:
                data = LocomoEvalService(self._request_config()).get_run(parts[-1])
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            self._send_json(HTTPStatus.OK, data)
        except LocomoEvalError as exc:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "locomo_eval_error", "message": str(exc)})

    def _agent_account_ensure(self) -> None:
        try:
            payload = self._read_json()
            label = str(payload.get("label") or "locomo").strip() or "locomo"
            data = self._ensure_account(label)
            self._send_json(HTTPStatus.OK, data)
        except (ValueError, OSError, EchoMemoryClientError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "account_ensure_error", "message": str(exc)})

    def _agent_account_create(self) -> None:
        try:
            payload = self._read_json()
            label = str(payload.get("label") or "").strip()
            if not label:
                raise ValueError("label is required")
            data = _create_account_record(self.server.config, self.server.echomem_url, label)
            self._send_json(HTTPStatus.OK, data)
        except (ValueError, OSError, EchoMemoryClientError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "account_create_error", "message": str(exc)})

    def _agent_account_forget(self) -> None:
        try:
            payload = self._read_json()
            label = str(payload.get("label") or "").strip()
            account_id = str(payload.get("id") or "").strip()
            data = _forget_account_record(self.server.config, label=label, account_id=account_id)
            self._send_json(HTTPStatus.OK, data)
        except (ValueError, OSError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "account_forget_error", "message": str(exc)})

    def _ensure_account(self, label: str) -> dict[str, Any]:
        return _ensure_account_record(self.server.config, self.server.echomem_url, label)

    def _echomem_json(self, path: str, *, method: str = "GET", payload: dict[str, Any] | None = None) -> dict[str, Any]:
        body = json.dumps(payload or {}, ensure_ascii=False).encode("utf-8") if method != "GET" else None
        request = Request(
            urljoin(self.server.echomem_url, path.lstrip("/")),
            data=body,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=10) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise EchoMemoryClientError(f"echomemory_http_{exc.code}: {detail}") from exc
        except URLError as exc:
            raise EchoMemoryClientError(f"echomemory_unreachable: {exc.reason}") from exc
        if not isinstance(data, dict):
            raise EchoMemoryClientError("EchoMemory response must be a JSON object")
        return data

    def _request_config(self) -> AgentConfig:
        auth_key = self.headers.get("X-Auth-Key", "").strip()
        if not auth_key:
            return self.server.config
        return replace(
            self.server.config,
            echomemory=replace(self.server.config.echomemory, auth_key=auth_key),
        )

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
    server = AgentPlaygroundServer((host, port), AgentRequestHandler, config=config)
    try:
        _ensure_account_record(config, server.echomem_url, "locomo", validate_cached=False)
    except Exception as exc:  # noqa: BLE001 - server can still run without preseeded account.
        print(f"warning: failed to preseed locomo account: {exc}")
    return server


def _read_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _ensure_account_record(
    config: AgentConfig,
    echomem_url: str,
    label: str,
    *,
    validate_cached: bool = True,
) -> dict[str, Any]:
    registry_path = Path(config.locomo.data_dir) / "accounts.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry = _read_registry(registry_path)
    if label in registry and registry[label].get("authKey"):
        if not validate_cached:
            return {"created": False, "account": registry[label]}
        if _account_auth_key_is_valid(echomem_url, str(registry[label].get("authKey") or "")):
            return {"created": False, "account": registry[label]}
        registry.pop(label, None)
        registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    return _create_account_record(config, echomem_url, label, registry=registry)


def _account_auth_key_is_valid(echomem_url: str, auth_key: str) -> bool:
    if not auth_key:
        return False
    request = Request(
        urljoin(echomem_url, "/agent/inspect/events"),
        method="GET",
        headers={"X-Auth-Key": auth_key},
    )
    try:
        with urlopen(request, timeout=10):
            return True
    except HTTPError as exc:
        if exc.code == HTTPStatus.UNAUTHORIZED:
            exc.read()
            return False
        detail = exc.read().decode("utf-8", errors="replace")
        raise EchoMemoryClientError(f"echomemory_http_{exc.code}: {detail}") from exc
    except URLError as exc:
        raise EchoMemoryClientError(f"echomemory_unreachable: {exc.reason}") from exc
    except TimeoutError as exc:
        raise EchoMemoryClientError(f"echomemory_timeout: {exc}") from exc


def _create_account_record(
    config: AgentConfig,
    echomem_url: str,
    label: str,
    *,
    registry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not label or "/" in label or "\\" in label:
        raise ValueError("label must be a non-empty name without path separators")
    registry_path = Path(config.locomo.data_dir) / "accounts.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry = registry if registry is not None else _read_registry(registry_path)
    tenant = _echomem_json_url(echomem_url, "/api/auth/tenants", method="POST", payload={"name": label})
    tenant_id = str((tenant.get("tenant") or {}).get("tenant_id") or "")
    if not tenant_id:
        raise ValueError("tenant creation returned no id")
    user = _echomem_json_url(echomem_url, f"/api/auth/tenants/{tenant_id}/users", method="POST", payload={})
    user_id = str((user.get("user") or {}).get("user_id") or "")
    if not user_id:
        raise ValueError("user creation returned no id")
    key = _echomem_json_url(echomem_url, f"/api/auth/tenants/{tenant_id}/users/{user_id}/key", method="POST", payload={})
    auth_key = str(key.get("auth_key") or "")
    if not auth_key:
        raise ValueError("auth key creation returned no key")

    account = {"id": tenant_id, "label": label, "tenantId": tenant_id, "userId": user_id, "authKey": auth_key}
    key = label
    if key in registry:
        suffix = 2
        while f"{label} {suffix}" in registry:
            suffix += 1
        key = f"{label} {suffix}"
        account["label"] = key
    registry[key] = account
    registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"created": True, "account": account}


def _forget_account_record(config: AgentConfig, *, label: str, account_id: str) -> dict[str, Any]:
    registry_path = Path(config.locomo.data_dir) / "accounts.json"
    registry = _read_registry(registry_path)
    removed = []
    for key, account in list(registry.items()):
        if (label and key == label) or (account_id and str(account.get("id") or "") == account_id):
            removed.append(account)
            registry.pop(key, None)
    if registry_path.exists():
        registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": "forgotten", "removed": removed}


def _echomem_json_url(echomem_url: str, path: str, *, method: str = "GET", payload: dict[str, Any] | None = None) -> dict[str, Any]:
    body = json.dumps(payload or {}, ensure_ascii=False).encode("utf-8") if method != "GET" else None
    request = Request(
        urljoin(echomem_url, path.lstrip("/")),
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise EchoMemoryClientError(f"echomemory_http_{exc.code}: {detail}") from exc
    except URLError as exc:
        raise EchoMemoryClientError(f"echomemory_unreachable: {exc.reason}") from exc
    if not isinstance(data, dict):
        raise EchoMemoryClientError("EchoMemory response must be a JSON object")
    return data


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

