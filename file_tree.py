import os
import graphviz

def build_graph_with_ellipsis(graph, parent_node_id, start_path, max_files_per_dir=3, current_depth=0, max_depth=3):
    """
    递归遍历文件夹结构，限制文件数量，并将结构添加到 Graphviz 图中。
    """
    if current_depth > max_depth:
        return

    try:
        entries = os.listdir(start_path)
    except PermissionError:
        error_node_id = f"err_{parent_node_id}_{current_depth}"
        graph.node(error_node_id, label="![权限错误]", style="filled", fillcolor="red")
        graph.edge(parent_node_id, error_node_id, arrowhead="none")
        return
    except FileNotFoundError:
        error_node_id = f"err_{parent_node_id}_{current_depth}"
        graph.node(error_node_id, label="![文件夹未找到]", style="filled", fillcolor="red")
        graph.edge(parent_node_id, error_node_id, arrowhead="none")
        return
    except Exception as e:
        error_node_id = f"err_{parent_node_id}_{current_depth}"
        graph.node(error_node_id, label=f"![访问错误: {e}]", style="filled", fillcolor="red")
        graph.edge(parent_node_id, error_node_id, arrowhead="none")
        return

    entries = sorted(entries, key=lambda x: (not os.path.isdir(os.path.join(start_path, x)), x))
    dirs = [e for e in entries if os.path.isdir(os.path.join(start_path, e))]
    files = [e for e in entries if os.path.isfile(os.path.join(start_path, e))]

    ellipsis_count = 0
    display_files = files[:max_files_per_dir]
    ellipsis_count = max(0, len(files) - max_files_per_dir)
    
    display_entries = dirs + display_files
    
    for entry_name in display_entries:
        path = os.path.join(start_path, entry_name)
        is_dir = os.path.isdir(path)
        
        # 创建唯一节点 ID
        node_id = path.replace(os.path.sep, '_').replace('.', '_')
        
        # 创建子节点
        label = entry_name + ("/" if is_dir else "")
        style = "filled" if is_dir else ""
        fillcolor = "lightblue" if is_dir else "white"
        
        graph.node(node_id, label=label, shape="box", style=style, fillcolor=fillcolor)
        
        # 创建父节点到子节点的边，并指定箭头类型为 none
        graph.edge(parent_node_id, node_id, arrowhead="none")

        # 递归调用
        if is_dir:
            build_graph_with_ellipsis(graph, node_id, path, max_files_per_dir, current_depth + 1, max_depth)

    # 如果文件数量超过限制，在最后添加省略号节点
    if ellipsis_count > 0:
        ellipsis_node_id = f"ellipsis_{parent_node_id}_{current_depth}"
        ellipsis_label = f"... ({ellipsis_count} more files omitted)"
        
        graph.node(ellipsis_node_id, label=ellipsis_label, shape="note", style="dashed")
        graph.edge(parent_node_id, ellipsis_node_id, arrowhead="none")


def create_file_structure_plot(target_dir, file_limit, depth_limit, output_filename="file_structure_plot"):
    """主函数：设置 Graphviz 并启动构建过程"""
    if not os.path.isdir(target_dir):
        print(f"\n错误: {target_dir} 不是一个有效的目录路径。")
        return

    dot = graphviz.Digraph(comment='File Structure', format='png')
    dot.attr(rankdir='LR', size='20,10', resolution='300', ranksep='1.5', nodesep='0.5', splines='ortho')  # 调整大小和间距
    
    # 定义根节点
    root_name = os.path.basename(target_dir) or target_dir
    root_node_id = root_name.replace('.', '_')
    
    dot.node(root_node_id, label=root_name + "/", shape="box", style="filled", fillcolor="lightcoral")
    
    print(f"开始构建图，根目录: {root_name}")
    print(f"限制: 文件={file_limit}, 深度={depth_limit}")

    build_graph_with_ellipsis(dot, root_node_id, target_dir, file_limit, 0, depth_limit)

    try:
        dot.render(output_filename, view=True, cleanup=True)
        print(f"\n成功生成图表: {output_filename}.png")
        print("图表已尝试自动打开。")
    except graphviz.backend.ExecutableNotFound:
        print("\n错误: 找不到 Graphviz 可执行文件。请确保您已安装 Graphviz 软件并将其添加到系统 PATH 中。")
    except Exception as e:
        print(f"\n渲染图表时发生错误: {e}")

# =================================================================
# 运行配置
# =================================================================

# 替换为你的目标文件夹路径
target_dir = r"../My_DL_Project" 

# 每个目录下最多显示的文件数量
FILE_LIMIT = 3

# 最大递归深度 
DEPTH_LIMIT = 3

# 执行绘图函数
create_file_structure_plot(target_dir, FILE_LIMIT, DEPTH_LIMIT)
