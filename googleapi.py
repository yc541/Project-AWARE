import requests

api_key = 'xxxxxx'  # replace with a valid googlemap api key
base_url = "https://maps.googleapis.com/maps/api/staticmap?"
zoom = "20"
maptype = "satellite"
size = "600x600"


def google_getaerial(latlon):
    url = base_url + "center=" + latlon + "&zoom=" + zoom + "&maptype=" + maptype + "&size=" + size + "&key=" + api_key
    response = requests.get(url)
    with open('./mapsamples/testmap.png', 'wb') as file:   # save the image to .png file
        file.write(response.content)
    return


def google_avimarker(center_latlon, edge_latlons, avi_all):
    marker_url_all_t = '&markers=color:green%7Csize:tiny%7Clabel:P'
    marker_url_all_f = '&markers=color:red%7Csize:tiny%7Clabel:F'
    anyt = False
    anyf = False
    for edge_latlon, avi in zip(edge_latlons, avi_all):
        if avi == True:
            anyt = True
            marker_url_all_t = marker_url_all_t + '%7C' + edge_latlon
        else:
            anyf = True
            marker_url_all_f = marker_url_all_f + '%7C' + edge_latlon
    if anyt == False:
        marker_url_all_t = ''
    if anyf == False:
        marker_url_all_f = ''

    url = base_url + "center=" + center_latlon + "&zoom=" + zoom + "&maptype=" + \
          maptype + "&size=" + size + marker_url_all_t + marker_url_all_f +\
          "&key=" + api_key
    response = requests.get(url)
    with open('./mapsamples/testmap_avi.png', 'wb') as file:  # save the image to .png file
        file.write(response.content)
    return
