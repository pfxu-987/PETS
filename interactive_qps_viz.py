import re
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 读取日志文件
def read_log_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    return lines

# 解析日志行
def parse_log_line(line):
    pattern = r'total_queries:(\d+),task_num:(\d+),stream_num:(\d+),batch_size:(\d+),seq_len:(\d+) QPS: ([\d\.]+)'
    match = re.match(pattern, line)
    if match:
        return {
            'total_queries': int(match.group(1)),
            'task_num': int(match.group(2)),
            'stream_num': int(match.group(3)),
            'batch_size': int(match.group(4)),
            'seq_len': int(match.group(5)),
            'qps': float(match.group(6))
        }
    return None

# 解析所有日志行并创建DataFrame
def create_dataframe(lines):
    data = []
    for line in lines:
        parsed = parse_log_line(line)
        if parsed:
            data.append(parsed)
    return pd.DataFrame(data)

# 主函数
def main():
    log_file = "research/exp_results/pets/bert_base/4.16_2.log"
    
    # 检查文件是否存在
    if not os.path.exists(log_file):
        print(f"文件 {log_file} 不存在!")
        return
    
    lines = read_log_file(log_file)
    df = create_dataframe(lines)
    
    # 获取唯一参数值
    stream_nums = sorted(df['stream_num'].unique())
    batch_sizes = sorted(df['batch_size'].unique())
    seq_lens = sorted(df['seq_len'].unique())
    
    # 选择默认值
    default_batch_size = 128
    default_seq_len = 16
    default_stream_num = 8
    
    # 创建一个增强的交互控件 - 完全重新设计
    fig_interactive = go.Figure()
    
    # 初始状态：x轴为batch_size，按照seq_len分组，固定stream_num=8
    x_param = 'batch_size'
    color_param = 'seq_len'
    fixed_param = 'stream_num'
    fixed_value = 8
    
    # 添加每个seq_len的曲线
    for color_value in (seq_lens if color_param == 'seq_len' else 
                      (batch_sizes if color_param == 'batch_size' else stream_nums)):
        # 构建过滤条件
        filter_cond = {fixed_param: fixed_value, color_param: color_value}
        temp_df = df
        for param, value in filter_cond.items():
            temp_df = temp_df[temp_df[param] == value]
        
        # 排序并添加数据
        temp_df = temp_df.sort_values(x_param)
        
        # 设置名称标签
        name = f"{color_param}={color_value}"
        
        fig_interactive.add_trace(go.Scatter(
            x=temp_df[x_param],
            y=temp_df['qps'],
            mode='lines+markers',
            marker=dict(size=8),
            line=dict(width=2),
            name=name
        ))
    
    # 更新初始布局
    fig_interactive.update_layout(
        xaxis=dict(
            title=x_param.replace('_', ' ').title(),
            type='linear'
        ),
        yaxis=dict(
            title='QPS',
            type='linear'
        ),
        title=f'QPS vs {x_param.replace("_", " ").title()} (Grouped by {color_param.replace("_", " ").title()}, {fixed_param}={fixed_value})',
        hovermode='closest',
        legend=dict(
            title=f"{color_param.replace('_', ' ').title()}",
            font=dict(size=10),
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(0, 0, 0, 0.2)',
            borderwidth=1
        ),
        width=1000,
        height=600,
        margin=dict(t=150)  # 增加顶部空间，为下拉菜单腾出位置
    )
    
    # 创建更新图表的函数
    def create_update_traces(x, color, fixed, value):
        # 添加所有可能的曲线
        traces = []
        for color_value in (seq_lens if color == 'seq_len' else 
                          batch_sizes if color == 'batch_size' else stream_nums):
            # 构建过滤条件
            filter_cond = {fixed: value, color: color_value}
            temp_df = df.copy()
            for param, val in filter_cond.items():
                temp_df = temp_df[temp_df[param] == val]
            
            # 排序
            temp_df = temp_df.sort_values(x)
            
            # 确保有数据
            if not temp_df.empty:
                traces.append(
                    dict(
                        type='scatter',
                        x=temp_df[x],
                        y=temp_df['qps'],
                        mode='lines+markers',
                        name=f"{color}={color_value}"
                    )
                )
        return traces
    
    # 创建X轴参数选择按钮
    x_buttons = [
        dict(
            method='update',
            label=f'X: {param.replace("_", " ").title()}',
            args=[
                {'visible': [True] * len(fig_interactive.data)},  # 先保持可见性
                {
                    'xaxis.title': param.replace('_', ' ').title(),
                    'title': f'QPS vs {param.replace("_", " ").title()} (Grouped by {color_param.replace("_", " ").title()}, {fixed_param}={fixed_value})'
                }
            ]
        ) for param in ['batch_size', 'seq_len', 'stream_num']
    ]
    
    # 创建颜色分组参数选择按钮
    color_buttons = [
        dict(
            method='update',
            label=f'Group by: {param.replace("_", " ").title()}',
            args=[
                {'visible': [True] * len(fig_interactive.data)},  # 先保持可见性
                {
                    'title': f'QPS vs {x_param.replace("_", " ").title()} (Grouped by {param.replace("_", " ").title()}, {fixed_param}={fixed_value})',
                    'legend.title': param.replace('_', ' ').title()
                }
            ]
        ) for param in ['batch_size', 'seq_len', 'stream_num'] if param != fixed_param
    ]
    
    # 创建固定参数选择按钮
    fixed_buttons = [
        dict(
            method='update',
            label=f'Fixed: {param.replace("_", " ").title()}',
            args=[
                {'visible': [True] * len(fig_interactive.data)},  # 先保持可见性
                {
                    'title': f'QPS vs {x_param.replace("_", " ").title()} (Grouped by {color_param.replace("_", " ").title()}, {param}={fixed_value})'
                }
            ]
        ) for param in ['batch_size', 'seq_len', 'stream_num'] 
        if param != x_param and param != color_param
    ]
    
    # 添加复合更新按钮 (处理完整的图表重建)
    complex_buttons = []
    
    # 组合所有可能的x轴，颜色参数和固定参数
    param_options = ['batch_size', 'seq_len', 'stream_num']
    for x in param_options:
        for fixed in param_options:
            if x == fixed:
                continue
            
            for color in param_options:
                if color == x or color == fixed:
                    continue
                
                # 对于每个固定参数可能的值
                for fixed_val in (batch_sizes if fixed == 'batch_size' else
                                seq_lens if fixed == 'seq_len' else stream_nums):
                    # 创建更新
                    new_traces = create_update_traces(x, color, fixed, fixed_val)
                    
                    # 只添加合理的组合
                    if len(new_traces) > 0:
                        complex_buttons.append(
                            dict(
                                method='update',
                                label=f'X: {x.title()}, Color: {color.title()}, {fixed.title()}={fixed_val}',
                                args=[
                                    {'visible': [False] * len(fig_interactive.data)},  # 隐藏所有
                                    {
                                        'xaxis.title': x.replace('_', ' ').title(),
                                        'title': f'QPS vs {x.replace("_", " ").title()} (Grouped by {color.replace("_", " ").title()}, {fixed}={fixed_val})',
                                        'legend.title': color.replace('_', ' ').title()
                                    }
                                ]
                            )
                        )
    
    # 添加下拉菜单
    fig_interactive.update_layout(
        updatemenus=[
            # 添加综合下拉菜单 - 一次性选择三个参数
            {
                'buttons': complex_buttons[:20],  # 限制数量以防止菜单过长
                'direction': 'down',
                'showactive': True,
                'x': 0.1,
                'y': 1.15,
                'xanchor': 'left',
                'yanchor': 'top',
                'pad': {'l': 10, 'r': 10, 't': 10, 'b': 10},
                'bgcolor': 'rgba(200, 200, 200, 0.8)',
                'font': {'size': 10},
                'name': 'Parameter Combination'
            }
        ]
    )
    
    # 创建一个全新的交互式视图 - 使用plotly express更简单地实现
    # 这个新的视图将使用下拉菜单让用户选择要绘制的参数
    fig_combined = go.Figure()
    
    # 默认配置：x=batch_size, color=seq_len, fixed=stream_num, fixed_value=8
    for seq_len in seq_lens:
        filtered_df = df[(df['stream_num'] == 8) & (df['seq_len'] == seq_len)]
        filtered_df = filtered_df.sort_values('batch_size')
        
        fig_combined.add_trace(go.Scatter(
            x=filtered_df['batch_size'],
            y=filtered_df['qps'],
            mode='lines+markers',
            name=f'seq_len={seq_len}',
            marker={'size': 8}
        ))
    
    # 创建下拉菜单选项
    dropdown_options = []
    
    # 1. X轴为batch_size，颜色为seq_len，固定stream_num
    for stream_num in stream_nums:
        visible = [False] * len(fig_combined.data)
        traces_added = 0
        
        for seq_len in seq_lens:
            filtered_df = df[(df['stream_num'] == stream_num) & (df['seq_len'] == seq_len)]
            if not filtered_df.empty:
                if traces_added < len(visible):
                    visible[traces_added] = True
                traces_added += 1
        
        dropdown_options.append({
            'method': 'update',
            'label': f'X: Batch Size, Color: Seq Len, Stream Num={stream_num}',
            'args': [
                {'visible': visible},
                {'title': f'QPS vs Batch Size (Grouped by Sequence Length, Stream Num={stream_num})'}
            ]
        })
    
    # 2. X轴为batch_size，颜色为stream_num，固定seq_len
    for seq_len in seq_lens:
        visible = [False] * len(fig_combined.data)
        new_traces = []
        
        for stream_num in stream_nums:
            filtered_df = df[(df['stream_num'] == stream_num) & (df['seq_len'] == seq_len)]
            filtered_df = filtered_df.sort_values('batch_size')
            
            if not filtered_df.empty:
                new_traces.append(go.Scatter(
                    x=filtered_df['batch_size'],
                    y=filtered_df['qps'],
                    mode='lines+markers',
                    name=f'stream_num={stream_num}',
                    marker={'size': 8}
                ))
        
        dropdown_options.append({
            'method': 'restyle',
            'label': f'X: Batch Size, Color: Stream Num, Seq Len={seq_len}',
            'args': [
                {'visible': visible + [True] * len(new_traces), 'x': [t.x for t in new_traces], 'y': [t.y for t in new_traces], 'name': [t.name for t in new_traces]},
                [i for i in range(len(visible))]
            ]
        })
    
    # 添加下拉菜单
    fig_combined.update_layout(
        updatemenus=[
            {
                'buttons': dropdown_options[:20],  # 限制数量以避免菜单过长
                'direction': 'down',
                'showactive': True,
                'x': 0.1,
                'y': 1.15,
                'xanchor': 'left',
                'yanchor': 'top',
                'pad': {'l': 10, 'r': 10, 't': 10, 'b': 10},
                'bgcolor': 'rgba(200, 200, 200, 0.8)',
                'font': {'size': 10}
            }
        ],
        title='Interactive Parameter Comparison',
        xaxis_title='Batch Size',
        yaxis_title='QPS',
        hovermode='closest',
        legend=dict(
            title='Sequence Length',
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(0, 0, 0, 0.2)',
            borderwidth=1
        ),
        width=1000,
        height=600,
        margin=dict(t=150)  # 增加顶部空间，为下拉菜单腾出位置
    )
    
    # 启用缩放和拖动
    fig_combined.update_layout(
        dragmode='zoom',
        hoverlabel=dict(
            bgcolor="white",
            font_size=14
        )
    )
    
    # 创建序列长度vs QPS图表（具有放大功能）
    fig_seq = px.line(df, x='seq_len', y='qps', color='batch_size', 
                 facet_col='stream_num', facet_col_wrap=3,
                 title='QPS vs Sequence Length (Grouped by batch_size and stream_num)',
                 labels={
                     'seq_len': 'Sequence Length',
                     'qps': 'QPS',
                     'batch_size': 'Batch Size',
                     'stream_num': 'Stream Number'
                 },
                 log_x=True)
    
    # 启用缩放和平移
    fig_seq.update_layout(
        dragmode='zoom',
        showlegend=True,
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        ),
        legend=dict(
            title_font_family="Times New Roman",
            font=dict(
                family="Courier",
                size=12,
                color="black"
            ),
            bgcolor="LightSteelBlue",
            bordercolor="Black",
            borderwidth=2
        )
    )
    
    # 创建批量大小vs QPS图表（具有放大功能）
    fig_batch = px.line(df, x='batch_size', y='qps', color='seq_len', 
                 facet_col='stream_num', facet_col_wrap=3,
                 title='QPS vs Batch Size (Grouped by seq_len and stream_num)',
                 labels={
                     'batch_size': 'Batch Size',
                     'qps': 'QPS',
                     'seq_len': 'Sequence Length',
                     'stream_num': 'Stream Number'
                 })
    
    # 启用缩放和平移
    fig_batch.update_layout(
        dragmode='zoom',
        showlegend=True,
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        ),
        legend=dict(
            title_font_family="Times New Roman",
            font=dict(
                family="Courier",
                size=12,
                color="black"
            ),
            bgcolor="LightSteelBlue",
            bordercolor="Black",
            borderwidth=2
        )
    )
    
    # 创建流数量vs QPS图表（具有放大功能）
    fig_stream = px.line(df, x='stream_num', y='qps', color='batch_size', 
                 facet_col='seq_len', facet_col_wrap=3,
                 title='QPS vs Stream Number (Grouped by batch_size and seq_len)',
                 labels={
                     'stream_num': 'Stream Number',
                     'qps': 'QPS',
                     'batch_size': 'Batch Size',
                     'seq_len': 'Sequence Length'
                 },
                 log_x=True)
    
    # 启用缩放和平移
    fig_stream.update_layout(
        dragmode='zoom',
        showlegend=True,
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        ),
        legend=dict(
            title_font_family="Times New Roman",
            font=dict(
                family="Courier",
                size=12,
                color="black"
            ),
            bgcolor="LightSteelBlue",
            bordercolor="Black",
            borderwidth=2
        )
    )
    
    # 保存所有图表到一个HTML文件
    with open('qps_interactive_analysis.html', 'w') as f:
        f.write(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>QPS Performance Analysis</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    max-width: 1800px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .chart-container {{
                    margin-bottom: 30px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 15px;
                }}
                h1 {{
                    color: #333;
                    text-align: center;
                }}
                h2 {{
                    color: #555;
                    margin-top: 20px;
                }}
                .nav {{
                    text-align: center;
                    margin: 20px 0;
                }}
                .nav a {{
                    margin: 0 10px;
                    padding: 8px 15px;
                    background-color: #f0f0f0;
                    border-radius: 5px;
                    text-decoration: none;
                    color: #333;
                }}
                .nav a:hover {{
                    background-color: #ddd;
                }}
                .fullscreen-btn {{
                    float: right;
                    margin: 10px;
                    padding: 5px 10px;
                    background-color: #007bff;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                }}
                .fullscreen-btn:hover {{
                    background-color: #0056b3;
                }}
                .chart-container {{
                    position: relative;
                }}
            </style>
            <script>
                function toggleFullScreen(elementId) {{
                    var element = document.getElementById(elementId);
                    
                    if (!document.fullscreenElement) {{
                        if (element.requestFullscreen) {{
                            element.requestFullscreen();
                        }} else if (element.mozRequestFullScreen) {{ /* Firefox */
                            element.mozRequestFullScreen();
                        }} else if (element.webkitRequestFullscreen) {{ /* Chrome, Safari & Opera */
                            element.webkitRequestFullscreen();
                        }} else if (element.msRequestFullscreen) {{ /* IE/Edge */
                            element.msRequestFullscreen();
                        }}
                    }} else {{
                        if (document.exitFullscreen) {{
                            document.exitFullscreen();
                        }} else if (document.mozCancelFullScreen) {{
                            document.mozCancelFullScreen();
                        }} else if (document.webkitExitFullscreen) {{
                            document.webkitExitFullscreen();
                        }} else if (document.msExitFullscreen) {{
                            document.msExitFullscreen();
                        }}
                    }}
                }}
            </script>
        </head>
        <body>
            <h1>QPS Performance Analysis for BERT Model</h1>
            
            <div class="nav">
                <a href="#interactive">Interactive Analysis</a>
                <a href="#batch">Batch Size Analysis</a>
                <a href="#seq">Sequence Length Analysis</a>
                <a href="#stream">Stream Number Analysis</a>
            </div>
            
            <h2 id="interactive">Interactive Parameter Analysis</h2>
            <div class="chart-container">
                <button class="fullscreen-btn" onclick="toggleFullScreen('interactive-chart')">⤢ Fullscreen</button>
                <div id="interactive-chart"></div>
            </div>
            
            <h2 id="batch">QPS vs Batch Size</h2>
            <div class="chart-container">
                <button class="fullscreen-btn" onclick="toggleFullScreen('batch-chart')">⤢ Fullscreen</button>
                <div id="batch-chart"></div>
            </div>
            
            <h2 id="seq">QPS vs Sequence Length</h2>
            <div class="chart-container">
                <button class="fullscreen-btn" onclick="toggleFullScreen('seq-chart')">⤢ Fullscreen</button>
                <div id="seq-chart"></div>
            </div>
            
            <h2 id="stream">QPS vs Stream Number</h2>
            <div class="chart-container">
                <button class="fullscreen-btn" onclick="toggleFullScreen('stream-chart')">⤢ Fullscreen</button>
                <div id="stream-chart"></div>
            </div>
            
            <script>
                var fig_combined = {fig_combined.to_json()};
                Plotly.newPlot('interactive-chart', fig_combined.data, fig_combined.layout);
                
                var fig_batch = {fig_batch.to_json()};
                Plotly.newPlot('batch-chart', fig_batch.data, fig_batch.layout);
                
                var fig_seq = {fig_seq.to_json()};
                Plotly.newPlot('seq-chart', fig_seq.data, fig_seq.layout);
                
                var fig_stream = {fig_stream.to_json()};
                Plotly.newPlot('stream-chart', fig_stream.data, fig_stream.layout);
            </script>
        </body>
        </html>
        """)
    
    print("交互式QPS分析HTML文件已生成: qps_interactive_analysis.html")
    print("请在浏览器中打开该文件进行查看和交互")

if __name__ == "__main__":
    main() 