import classify as cl
import gradio as gr

def greet(file):
    cl.savefig(cl.get_clusters(cl.process_data(file)), "fig.png")
    return "fig.png"

demo = gr.Interface(
    fn=greet,
    inputs=["file"],
    outputs=["image"],
)

demo.launch()
