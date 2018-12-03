"""
    箱型图绘制
"""
import plotly.plotly
import plotly.graph_objs as go

data = [
    go.Box(
        y=[1, 2, 4, 3, 4, 5, 6, 7, 7, 9]
    )
]
plotly.offline.plot(data)