import requests
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

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


def get_order_status(id):
    s = get_session()
    response = s.get(usgs_host+'/order-status/'+id)
    response.raise_for_status()
    data = response.json()
    return data['status']

# copied directly from the epsg code
valid_statuses = ['complete', 'queued', 'oncache', 'onorder', 'purged',
                      'processing', 'error', 'unavailable', 'submitted']
all_but_purged = valid_statuses[:]
all_but_purged.remove('purged')

def get_open_orders():
    """Get the list and status of all orders that have not been purged"""
    s = get_session()
    response = s.get(usgs_host+'/list-orders', json={"status": all_but_purged })
    response.raise_for_status()
    return response.json()


def download_file(url,saveas):
    """Download a file, saving it in the path requested.  Overwrites existing file."""
    s = get_session()
    response = s.get(url,stream=True)
    with open(saveas,'wb') as out:
        for chunk in response.iter_content(chunk_size=65535):
            out.write(chunk)

def process_item(item):
    """Take the information about a single item (data file) and decide what to do about it:
    ignore it (if it is not ready or if we already have it) or download it (if it is new and ready)"""
    if item['status'] == 'complete':
        file = Path(storage_directory) / (item['name'] + '.tar.gz')
        if not file.exists():
            print('starting download of ' + item['name'])
            download_file(item['product_dload_url'],file)
            return
    print("xxx ignoring " + item['name'] + " xxx")


def download_available_results():
    """Download any products that are ready and have not previously been
    downloaded."""
    s = get_session()
    orders = get_open_orders()
    with ThreadPoolExecutor(max_workers=5) as pool:
        # There's a double loop here: given each order name, fetch the status of all its items
        # for each item, download it (or not).  We keep the first loop on the main thread, for
        # simplicity, and use the thread pool for the actual downloads.
        for orderid in orders:
            response = s.get(usgs_host+'/item-status/'+orderid, json={"status": "complete"})
            response.raise_for_status()
            for item in response[orderid]:
                pool.submit(process_item,item)
    print("download complete")