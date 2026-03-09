import os
import sys

# Ensure the app directory is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.graph.workflow import create_workflow

def save_graph_image():
    """
    Visualizes the LangGraph workflow and saves it as agent_workflow.png.
    """
    print("Building workflow...")
    app = create_workflow()
    
    print("Generating graph visualization...")
    try:
        # LangGraph's get_graph() returns a drawable graph object
        # draw_mermaid_png() is the standard way to get PNG bytes (uses mermaid.ink)
        png_bytes = app.get_graph().draw_mermaid_png()
        
        output_path = "agent_workflow.png"
        with open(output_path, "wb") as f:
            f.write(png_bytes)
            
        print(f"Graph saved as {output_path}")
        
    except Exception as e:
        print(f"\n❌ Failed to generate graph: {e}")
        print("\nNote: draw_mermaid_png() requires internet access to call mermaid.ink.")
        print("If you are offline, you might need to use draw_png() which requires 'pygraphviz' and 'graphviz' installed.")

if __name__ == "__main__":
    save_graph_image()
