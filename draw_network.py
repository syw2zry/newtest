import graphviz

def draw_network_architecture():
    # 初始化有向图，设置输出格式为 PDF（最适合放入 LaTeX 或 Word 论文中）
    # splines='ortho' 表示使用正交折线，看起来更有工程图的严谨感
    dot = graphviz.Digraph('NetworkArchitecture', format='pdf')
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.6', ranksep='0.8')

    # 定义三种节点的全局样式
    # 1. 基础模块 (浅蓝色)
    base_style = {'style': 'filled,rounded', 'fillcolor': '#f0f8ff', 'color': '#a9d0f5', 'shape': 'box', 'fontname': 'Helvetica'}
    # 2. 核心创新模块 Ours (浅紫色，加粗边框)
    ours_style = {'style': 'filled,rounded', 'fillcolor': '#f9f0ff', 'color': '#d0a9f5', 'shape': 'box', 'fontname': 'Helvetica', 'penwidth': '2.0'}
    # 3. 输出节点 (浅绿色)
    out_style = {'style': 'filled', 'fillcolor': '#e6ffe6', 'color': '#a3e6a3', 'shape': 'ellipse', 'fontname': 'Helvetica-Bold', 'penwidth': '2.0'}
    
    # 边（箭头）的样式
    edge_style = {'fontname': 'Helvetica', 'fontsize': '10', 'fontcolor': '#555555'}

    # ================= Phase 1: 输入与基准特征提取 =================
    with dot.subgraph(name='cluster_phase1') as c:
        c.attr(label='Phase 1: Input & Base Feature Extraction', style='dashed', color='gray', fontname='Helvetica-Bold')
        
        c.node('L', 'Left Image', **base_style)
        c.node('R', 'Right Image', **base_style)
        
        c.node('S2L', 'Stem_2x (L)', **base_style)
        c.node('S2R', 'Stem_2x (R)', **base_style)
        c.node('S4L', 'Stem_4x (L)', **base_style)
        c.node('S4R', 'Stem_4x (R)', **base_style)
        
        c.node('FExt', 'Feature Extractor\n(MobileNetV2)', **base_style)
        c.node('FL', 'F_left (1/4)', **base_style)
        c.node('FR', 'F_right (1/4)', **base_style)

    dot.edge('L', 'S2L')
    dot.edge('R', 'S2R')
    dot.edge('S2L', 'S4L')
    dot.edge('S2R', 'S4R')
    dot.edge('L', 'FExt')
    dot.edge('R', 'FExt')
    dot.edge('FExt', 'FL')
    dot.edge('FExt', 'FR')

    # ================= Phase 2: 结构感知与频域引导 =================
    with dot.subgraph(name='cluster_phase2') as c:
        c.attr(label='Phase 2: Structural & Frequency Guidance (Ours)', style='dashed', color='#d0a9f5', fontname='Helvetica-Bold')
        
        # 边缘分支
        c.node('EGL', 'Edge Guidance (L)\n[Sobel + Sq Sigmoid]', **ours_style)
        c.node('EGR', 'Edge Guidance (R)\n[Sobel + Sq Sigmoid]', **ours_style)
        c.node('ML', 'Mask_L', **ours_style)
        c.node('MR', 'Mask_R', **ours_style)
        
        # 频域分支
        c.node('FDL', 'Freq Decoupler (L)', **ours_style)
        c.node('FDR', 'Freq Decoupler (R)', **ours_style)
        
        c.node('FL_low', 'F_low (L)', **ours_style)
        c.node('FL_high', 'F_high (L)', **ours_style)
        c.node('FR_low', 'F_low (R)', **ours_style)
        c.node('FR_high', 'F_high (R)', **ours_style)
        
        # 调制与重组节点
        c.node('FL_mod', 'F_high_modulated (L)', **ours_style)
        c.node('FR_mod', 'F_high_modulated (R)', **ours_style)
        c.node('ReL', 'F_refined (L)', **ours_style)
        c.node('ReR', 'F_refined (R)', **ours_style)
        c.node('MatchL', 'match_left', **ours_style)
        c.node('MatchR', 'match_right', **ours_style)

    # 连线 (Phase 1 到 Phase 2)
    dot.edge('L', 'EGL')
    dot.edge('R', 'EGR')
    dot.edge('EGL', 'ML')
    dot.edge('EGR', 'MR')
    
    dot.edge('FL', 'FDL')
    dot.edge('FR', 'FDR')
    dot.edge('FDL', 'FL_low')
    dot.edge('FDL', 'FL_high')
    dot.edge('FDR', 'FR_low')
    dot.edge('FDR', 'FR_high')
    
    # 核心张量运算标注
    dot.edge('FL_high', 'FL_mod', label='mul_ (x 1+Mask_L)', **edge_style)
    dot.edge('ML', 'FL_mod', style='dotted')
    dot.edge('FR_high', 'FR_mod', label='mul_ (x 1+Mask_R)', **edge_style)
    dot.edge('MR', 'FR_mod', style='dotted')
    
    dot.edge('FL_low', 'ReL', label='add_', **edge_style)
    dot.edge('FL_mod', 'ReL')
    dot.edge('FR_low', 'ReR', label='add_', **edge_style)
    dot.edge('FR_mod', 'ReR')
    
    dot.edge('ReL', 'MatchL', label='cat', **edge_style)
    dot.edge('S4L', 'MatchL')
    dot.edge('ReR', 'MatchR', label='cat', **edge_style)
    dot.edge('S4R', 'MatchR')

    # ================= Phase 3: 动态尺度聚合 =================
    with dot.subgraph(name='cluster_phase3') as c:
        c.attr(label='Phase 3: Adaptive Scale Cost Volume (Ours)', style='dashed', color='#d0a9f5', fontname='Helvetica-Bold')
        
        c.node('GWC', 'Group-Wise Correlation', **base_style)
        c.node('BaseVol', 'Base Cost Volume', **base_style)
        
        c.node('SP', 'Scale Predictor', **ours_style)
        c.node('W', 'Weights: W_s, W_m, W_l', **ours_style)
        
        c.node('AggS', '3D Agg (Dilation=1)', **ours_style)
        c.node('AggM', '3D Agg (Dilation=2)', **ours_style)
        c.node('AggL', '3D Agg (Dilation=4)', **ours_style)
        
        c.node('Fused', 'Fused Volume', **ours_style)
        c.node('AllDisp', 'all_disp_volume', **ours_style)

    # 连线 (Phase 2 到 Phase 3)
    dot.edge('MatchL', 'GWC')
    dot.edge('MatchR', 'GWC')
    dot.edge('GWC', 'BaseVol')
    
    dot.edge('MatchL', 'SP')
    dot.edge('ML', 'SP', style='dotted', label='Prior', **edge_style)
    dot.edge('SP', 'W')
    
    dot.edge('BaseVol', 'AggS')
    dot.edge('BaseVol', 'AggM')
    dot.edge('BaseVol', 'AggL')
    
    dot.edge('W', 'AggS', style='dotted')
    dot.edge('W', 'AggM', style='dotted')
    dot.edge('W', 'AggL', style='dotted')
    
    dot.edge('AggS', 'Fused', label='add_', **edge_style)
    dot.edge('AggM', 'Fused', label='add_', **edge_style)
    dot.edge('AggL', 'Fused', label='add_', **edge_style)
    dot.edge('Fused', 'AllDisp')

    # ================= Phase 4: 视差预测与迭代 =================
    with dot.subgraph(name='cluster_phase4') as c:
        c.attr(label='Phase 4: Disparity Regression & Refinement', style='dashed', color='gray', fontname='Helvetica-Bold')
        
        c.node('Hourglass', '3D Hourglass', **base_style)
        c.node('Disp0', 'disp_0 (Initial)', **base_style)
        
        c.node('GRU', 'ConvGRU Update\n(iters=22)', **base_style)
        c.node('DispN', 'disp_i+1', **base_style)
        
        c.node('Up', 'Spatial Upsample', **base_style)
        c.node('Final', 'Final Disparity Map', **out_style)

    # 连线 (Phase 3 到 Phase 4)
    dot.edge('AllDisp', 'Hourglass')
    dot.edge('Hourglass', 'Disp0')
    dot.edge('Disp0', 'GRU')
    dot.edge('L', 'GRU', style='dotted', label='Context', **edge_style)
    dot.edge('GRU', 'DispN')
    dot.edge('DispN', 'GRU', label='Loop', **edge_style) # 形成环路表示迭代
    
    dot.edge('DispN', 'Up')
    dot.edge('S2L', 'Up', style='dotted') # 依据代码，upsample 用到了 stem_2x
    dot.edge('Up', 'Final')

    # 直接打印底层 Dot 源码
    print("请复制下面 ---------- 之间的所有代码：")
    print("-" * 50)
    print(dot.source)
    print("-" * 50)

if __name__ == '__main__':
    draw_network_architecture()