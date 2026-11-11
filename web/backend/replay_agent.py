"""Independent source worker. Receives ordered replay assignments and forwards logs."""
import os
import time

import httpx

DEFAULT_URL = 'http://127.0.0.1:8000'
REQUEST_TIMEOUT = 10
POLL_INTERVAL = 1.0
MIN_SLEEP = 0.05
RETRY_DELAY = 2
CONFLICT = 409  # the backend already holds a newer cursor for this source


def forward_work(client, url, source):
    """Fetch the next assignment for this source and post its events back to the backend."""
    response = client.get(f'{url}/api/sources/{source}/work')
    response.raise_for_status()
    work = response.json()
    if not work['events']:
        return
    response = client.post(f'{url}/api/sources/{source}/ingest',
                           json={'session_id': work['session_id'], 'cursor': work['cursor'],
                                 'events': work['events']})
    if response.status_code != CONFLICT:
        response.raise_for_status()


def main():
    source = os.environ['SOURCE_ID']
    url = os.environ.get('OBSERVATORY_URL', DEFAULT_URL).rstrip('/')
    headers = {'Authorization': 'Bearer ' + os.environ['SOURCE_TOKEN']}
    with httpx.Client(timeout=REQUEST_TIMEOUT, headers=headers) as client:
        while True:
            started = time.monotonic()
            try:
                forward_work(client, url, source)
            except httpx.HTTPError:
                # Cursor is owned by the backend. Retrying after reconnect is idempotent.
                time.sleep(RETRY_DELAY)
            time.sleep(max(MIN_SLEEP, POLL_INTERVAL - (time.monotonic() - started)))


if __name__ == '__main__':
    main()
