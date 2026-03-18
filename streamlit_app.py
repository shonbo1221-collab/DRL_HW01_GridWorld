import streamlit as st
import numpy as np
from rl_env import evaluate_policy, value_iteration

st.set_page_config(page_title="HW1: GridWorld RL", layout="wide")

st.markdown("""
<style>
    div[data-testid="column"] { padding: 0 !important; }
    .stButton>button { width: 100%; height: 60px; font-size: 24px; padding: 0 !important;}
</style>
""", unsafe_allow_html=True)

st.title("HW1: GridWorld 強化學習 (Streamlit 版)")

if 'n' not in st.session_state: st.session_state.n = 5
if 'click_count' not in st.session_state: st.session_state.click_count = 0
if 'start_idx' not in st.session_state: st.session_state.start_idx = -1
if 'end_idx' not in st.session_state: st.session_state.end_idx = -1
if 'obstacles' not in st.session_state: st.session_state.obstacles = []
if 'eval_done' not in st.session_state: st.session_state.eval_done = False
if 'results' not in st.session_state: st.session_state.results = {}

st.sidebar.header("控制面板")
n_input = st.sidebar.number_input("維度 n (5~9)", min_value=5, max_value=9, value=st.session_state.n)

if st.sidebar.button("生成/重設 網格", type="primary"):
    st.session_state.n = n_input
    st.session_state.click_count = 0
    st.session_state.start_idx = -1
    st.session_state.end_idx = -1
    st.session_state.obstacles = []
    st.session_state.eval_done = False
    st.rerun()

total_obs = st.session_state.n - 2

if st.session_state.click_count == 0:
    st.info("請點擊下方格子設定「起點」(綠色🟩)")
elif st.session_state.click_count == 1:
    st.warning("請點擊下方格子設定「終點」(紅色🟥)")
elif st.session_state.click_count >= 2 and st.session_state.click_count < 2 + total_obs:
    rem = (2 + total_obs) - st.session_state.click_count
    st.error(f"請點擊下方格子設定「障礙物」(黑色⬛)，剩餘 {rem} 個")
else:
    st.success("設定完成！可以點擊下方「執行 (評估與迭代)」按鈕開始運算。")

def handle_click(i):
    if i == st.session_state.start_idx or i == st.session_state.end_idx or i in st.session_state.obstacles:
        return
    if st.session_state.click_count == 0:
        st.session_state.start_idx = i
        st.session_state.click_count += 1
    elif st.session_state.click_count == 1:
        st.session_state.end_idx = i
        st.session_state.click_count += 1
    elif st.session_state.click_count < 2 + total_obs:
        st.session_state.obstacles.append(i)
        st.session_state.click_count += 1

n = st.session_state.n
st.write("### 網格設定區 (點擊格子)")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    for r in range(n):
        cols = st.columns(n)
        for c in range(n):
            i = r * n + c
            label = "⬜"
            if i == st.session_state.start_idx: label = "🟩"
            elif i == st.session_state.end_idx: label = "🟥"
            elif i in st.session_state.obstacles: label = "⬛"
            with cols[c]:
                st.button(label, key=f"cell_{i}", on_click=handle_click, args=(i,))

st.write("")
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    if st.session_state.click_count >= 2 + total_obs:
        if st.button("🚀 執行 (評估與迭代)", type="primary", use_container_width=True):
            with st.spinner("運算中..."):
                pol1, val1 = evaluate_policy(n, st.session_state.end_idx, st.session_state.obstacles)
                pol2, val2 = value_iteration(n, st.session_state.end_idx, st.session_state.obstacles)
                st.session_state.results = {'eval': (pol1, val1), 'val_iter': (pol2, val2)}
                st.session_state.eval_done = True

def find_path(policy, start, end, obstacles, n):
    path = []
    current = start
    visited = set()
    if start == -1 or end == -1: return path
    while current != end:
        if current in obstacles or current in visited: break
        arrow = policy.get(current, '')
        if not arrow: break
        path.append(current)
        visited.add(current)
        r, c = divmod(current, n)
        if arrow == '↑': r = max(0, r - 1)
        elif arrow == '↓': r = min(n - 1, r + 1)
        elif arrow == '←': c = max(0, c - 1)
        elif arrow == '→': c = min(n - 1, c + 1)
        nxt = r * n + c
        if nxt in obstacles: nxt = current
        current = nxt
    return path

def render_html_grid(title, policy, values, path_indices, n):
    html = f"<div style='flex:1; min-width:320px; padding:10px; font-family:sans-serif;'>"
    html += f"<h3 style='text-align:center; color:#2c3e50;'>{title}</h3>"
    html += f"<div style='display:grid; gap:2px; background-color:#ccc; border:2px solid #999; padding:2px; grid-template-columns:repeat({n},60px); grid-template-rows:repeat({n},60px); justify-content:center; margin:auto;'>"
    
    for i in range(n*n):
        bg = "white"
        color = "#333"
        content = ""
        
        if i in st.session_state.obstacles:
            bg = "#7f8c8d"
        elif i == st.session_state.end_idx:
            bg = "#e74c3c"
            color = "white"
            content = f"<span></span><br><span>{values[i]:.3f}</span>"
        else:
            if i == st.session_state.start_idx:
                bg = "#2ecc71"
                color = "white"
            elif i in path_indices:
                bg = "#f1c40f"
                color = "#333"
                
            arrow = policy.get(i, '')
            val = values[i]
            content = f"<span>{arrow}</span><br><span>{val:.3f}</span>"
            
        html += f"""
        <div style='background-color:{bg}; color:{color}; display:flex; flex-direction:column; align-items:center; justify-content:center; font-size:14px; font-weight:bold;'>
            <div style='text-align:center; line-height:1.2'>{content}</div>
        </div>
        """
    html += "</div></div>"
    return html

if st.session_state.eval_done:
    st.write("---")
    st.write("### 雙結果並列展示")
    pol1, val1 = st.session_state.results['eval']
    pol2, val2 = st.session_state.results['val_iter']
    path1 = find_path(pol1, st.session_state.start_idx, st.session_state.end_idx, st.session_state.obstacles, n)
    path2 = find_path(pol2, st.session_state.start_idx, st.session_state.end_idx, st.session_state.obstacles, n)
    
    h1 = render_html_grid("HW1-2 隨機策略評估", pol1, val1, path1, n)
    h2 = render_html_grid("HW1-3 最佳政策 (Value Iteration)", pol2, val2, path2, n)
    
    final_html = f"<div style='display:flex; flex-wrap:wrap; justify-content:space-around;'>{h1}{h2}</div>"
    st.components.v1.html(final_html, height=120 + n * 65)
