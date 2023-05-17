
import random
import io
import sys
from jinja2 import Template
import folium 
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QDateEdit, QPushButton
from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEnginePage 

import utils.show
from utils.process import get_mask, get_image
import matplotlib.pyplot as plt
import numpy 

class WebEnginePage(QWebEnginePage):
    def javaScriptConsoleMessage(self, level, message, lineNumber, sourceID):
        print("javaScriptConsoleMessage: ", level, message, lineNumber, sourceID)


def getLatLon(i, j):
    #return (54.290882 + (58.263287 - 54.290882) / 256 * i, 41.389207 + (48.151968 - 41.389207) / 256 * j, 54.290882 + (58.263287 - 54.290882) / 256 * (i + 1), 41.389207 + (48.151968 - 41.389207) / 256 * (j + 1))
    return (54.500526 + (55.263287 - 54.500526) / 256 * i, 41.389207 + (42.151968 - 41.389207) / 256 * j, 54.500526 + (55.263287 - 54.500526) / 256 * (i + 1), 41.389207 + (42.151968 - 41.389207) / 256 * (j + 1))


class MyApp(QMainWindow):

    def onLoad(self):
        js = Template("var f = true; var markerGroup = L.layerGroup().addTo({{map}});").render(map=self.map.get_name())
        self.webView.page().runJavaScript(js);
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Fire Segmentation')
        self.window_width, self.window_height = 900, 1000
        self.setMinimumSize(self.window_width, self.window_height)
        layout = QVBoxLayout()
        #self.setLayout(layout)
        coordinate = (56.296505, 43.936058)
        self.map = folium.Map(
            zoom_start=7, location=coordinate, control_scale=True, tiles=None
        )
        folium.raster_layers.TileLayer(
            tiles="http://mt1.google.com/vt/lyrs=m&h1=p1Z&x={x}&y={y}&z={z}",
            name="Standard Roadmap",
            attr="Google Map",
        ).add_to(self.map)
        folium.LayerControl().add_to(self.map)
        data = io.BytesIO()
        self.border = [[54.290882, 41.389207], [54.290882, 48.151968], [58.263287, 48.151968], [58.263287, 41.389207]]
        #folium.Polygon(locations = border, fill_color="orange").add_to(self.m)
        
        self.map.save(data, close_file=False)
        self.webView = QWebEngineView()
        self.webView.setPage(WebEnginePage(self.webView))
        self.webView.setHtml(data.getvalue().decode())
        layout.addWidget(self.webView)
        self.setCentralWidget(self.webView)

        self.webView.loadFinished.connect(self.onLoad)
        
        self.date_edit = QDateEdit(self)
        self.date_edit.setGeometry(100, 20, 120, 25)

        self.button = QPushButton('Set date', self)
        self.button.move(150,70)
        self.button.clicked.connect(self.update)
        

    def update(self):
        value = self.date_edit.date()
            
        print("done")
        print("{}T00:00:00Z".format(value.toString("yyyy-MM-dd")))
        print("done")
        time_interval = ("{}T00:00:00Z".format(value.toString("yyyy-MM-dd")), "2023-01-18T00:00:00Z")
        self.webView.page().runJavaScript("markerGroup.clearLayers();")
        bbox = [41.389207, 54.500526, 42.151968, 55.263287]
        y = get_mask(bbox, time_interval, "model-resnet50-novograd-0008.h5")
        print(y)
        k = numpy.size(y, 0)
        print(numpy.size(y, 0))
        print(numpy.size(y, 1))
        print(numpy.size(y, 2))
        
        for i in range(256):
            for j in range(256):
                for i1 in range(int(numpy.size(y, 0) / 256)):
                    for j1 in range(int(numpy.size(y, 1) / 256)):
                        #print(y[i * int(numpy.size(y, 0) / 256) + i1][j * int(numpy.size(y, 1) / 256) + j1])
                        if y[i * int(numpy.size(y, 0) / 256) + i1][j * int(numpy.size(y, 1) / 256) + j1][0] >= 0.00000000000001:
                            
                            js = Template(
                                """
                                L.polygon(
                                  [[{{firstLat}}, {{firstLon}}], [{{secondLat}}, {{secondLon}}], [{{thirdLat}}, {{thirdLon}}], [{{fourthLat}}, {{fourthLon}}]], fill_color='#fc0000', {weight: 0}).addTo(markerGroup);                """
                              #).render(map=self.map.get_name(), firstLat = self.border[0][0], firstLon = self.border[0][1],
                              #        secondLat = self.border[1][0], secondLon = self.border[1][1],
                              #        thirdLat = self.border[2][0], thirdLon = self.border[2][1],
                              #        fourthLat = self.border[3][0], fourthLon = self.border[3][1],
                              #        c1 = i, c2 = j)
                                ).render(firstLat = getLatLon(i, j)[0], firstLon = getLatLon(i, j)[1],
                                secondLat = getLatLon(i, j)[2], secondLon = getLatLon(i, j)[1],
                                thirdLat = getLatLon(i, j)[2], thirdLon = getLatLon(i, j)[3],
                                fourthLat = getLatLon(i, j)[0], fourthLon = getLatLon(i, j)[3],
                                c1 = i, c2 = j)
                            self.webView.page().runJavaScript(js)
                            break
                else:
                    continue
                break
                            
        
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet(''' QWidget { font-size: 20px; } ''')
    myApp = MyApp()
    myApp.show()
    try:
        sys.exit(app.exec_())
    except SystemExit:
        print('Closing Window...')




