import requests
import re


base_url = "https://wisdm.wisdmdevlaptop/api"
headers = {"X-API-Key": "xxxxx"}  # replace with a valid WISDM api key


def wisdm_getlatlon(postcode, address):
    postcode_url = "/availability/1/postcode-lookup?postcode=" + postcode
    response = requests.get(base_url+postcode_url, headers=headers, verify=False)
    content = response.content.decode('utf8')
    addr = '"' + address   # the " before address is needed for lookup
    idx_addr = content.find(addr)
    if idx_addr == -1:
        print(postcode, address)
        raise NameError("Address does not match postcode, please check")
    else:
        idx_lat = content.find('lat', idx_addr)
        lat = content[idx_lat + 5 : idx_lat + 14]
        idx_lon = content.find('lng', idx_addr)
        lon = content[idx_lon + 5 : idx_lon + 14]
    return lat + ', ' + lon


def wisdm_checkavi(latlon):
    index = latlon.index(',')
    lat = latlon[0:index]
    lon = latlon[index + 1:-1]
    check_url = "/availability/1/check?latitude=" + lat + "&longitude=" + lon
    response = requests.get(base_url + check_url, headers=headers, verify=False)
    content = response.content.decode('utf8')
    ap_idxs = [m.start() for m in re.finditer('name', content)]  # find all 'name' string in the response
    ap_idxs = ap_idxs[1::2]
    ap_names =[]
    avi_all = []
    for ap_idx in ap_idxs:
        idx1 = content.find('"', ap_idx+5)
        idx2 = content.find('"', idx1+1)    # idx 1 and 2 to find the string of ap name
        ap_name = content[idx1+1:idx2]
        ap_names.append(ap_name)
        idx3 = content.find('result', idx2)   # find the 'result' key word in response
        if content.find("pass",idx3, idx3 + 30) > 0:   # wisdm api returns nothing if there is an error
            avi = True
        else:
            avi = False
        avi_all.append(avi)
    if content.find("pass") > 0:  # wisdm api returns nothing if there is an error
        avi_comb = True
    else:
        avi_comb = False
    return ap_names, avi_all, avi_comb

