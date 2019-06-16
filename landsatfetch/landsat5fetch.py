import requests
import os
from pathlib import Path

usgs_host = 'https://espa.cr.usgs.gov/api/v0'
storage_directory = "/home/usgs/landsat5"

def default_auth():
    return (os.environ['USGS_USER'], os.environ['USGS_PASSWORD'])

def get_session(auth=None):
    if not hasattr(get_session,'s'):
        get_session.s = requests.Session()
        get_session.s.auth = auth if auth else default_auth()
    return get_session.s


def submit_order(list_of_ids):
    """Submit an order for the given list of landsat ids.
    For now, we are hardwiring all other parameters of the request (landsat5,
    sr, etc.)"""
    if len(list_of_ids) >= 5000:
        raise ValueError("Too many ids in a single request")
    # Build order body.  We aren't using any of the fancy options,
    # so this is quite simple
    order = {
        'tm5_collection': { 
            'inputs': list_of_ids,
            'products': ['sr']
        },
        'format': 'gtiff'
    }
    # send the order
    s = get_session()
    response = s.post(usgs_host+'/order',json=order)

    # error checking
    response.raise_for_status() # errors out if 404 etc.
    data = response.json()
    status = data['status'] if 'status' in data else 'unknown'
    if status != 'ordered':
        raise Exception('Order failed with status '+status)

    # return order id
    return data['orderid']


def get_order_status(order_id):
    s = get_session()
    response = s.get(usgs_host+'/order-status/'+order_id)
    response.raise_for_status()
    data = response.json()
    return data['status']

# copied directly from the epsg code... and wrong! (added 'ordered' status)
valid_statuses = ['ordered', 'complete', 'queued', 'oncache', 'onorder', 'purged',
                      'processing', 'error', 'unavailable', 'submitted']
all_but_purged = valid_statuses[:]
all_but_purged.remove('purged')

def get_open_orders():
    """Get the list and status of all orders that have not been purged"""
    s = get_session()
    response = s.get(usgs_host+'/list-orders', json={"status": all_but_purged })
    response.raise_for_status()
    return response.json()


def download_file(url, name, saveas):
    """Download a file, saving it in the path requested.  Overwrites existing file."""
    print('downloading ' + name, end=' ')
    s = get_session()
    response = s.get(url,stream=True)
    ctr = 0
    with open(saveas,'wb') as out:
       for chunk in response.iter_content(chunk_size=65535):
           out.write(chunk)
           ctr = ctr+1
           if ctr%50 == 0:
               print('.',end='')
    print('done')

skipped = []
def process_item(item):
    """Take the information about a single item (data file) and decide what to do about it:
    ignore it (if it is not ready or if we already have it) or download it (if it is new and ready)"""
    if item['status'] == 'complete':
        file = Path(storage_directory) / (item['name'] + '.tar.gz')
        if not file.exists():
            download_file(item['product_dload_url'], item['name'], file)
            return item['name']
        else:
            print(item['name'] + ': already have')
    else:
        print(item['name'] + ': not complete')
    skipped.append(item['name'])  # temporary debug
    return None


def download_available_results():
    """Download any products that are ready.  Returns a list of the downloaded names."""
    s = get_session()
    orders = get_open_orders()
    results = []
    # usgs doesn't like it if you download in parallel, so we don't any more
    for orderid in orders:
        print("order " + orderid)
        response = s.get(usgs_host+'/item-status/'+orderid, json={"status": "complete"})
        response.raise_for_status()
        data = response.json()
        for item in data[orderid]:
            r = process_item(item)
            if r:
                results.append(r)
    print("download complete")
    return results

def smudge(pw1: PixelWindow, pw2: PixelWindow) -> (PixelWindow, PixelWindow):
    """Due to rounding errors, it is possible that the same geo window results in pixel windows of two different sizes
    for different data files.  Smudge adjusts a matching pair of pixel windows so that they are definitely the same
    size.   Currently nothing clever here about geo registration; just making the height/width match."""
    common_height = min(pw1.height,pw2.height)
    common_width = min(pw1.width,pw2.width)
    if pw1.width != common_width or pw1.height != common_height:
        pw1 = rasterio.windows.Window(pw1.col_off,pw1.row_off,common_width,common_height)
    if pw2.width != common_width or pw2.height != common_height:
        pw2 = rasterio.windows.Window(pw2.col_off,pw2.row_off,common_width,common_height)
    return (pw1,pw2)