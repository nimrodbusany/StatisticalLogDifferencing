from URLMapper import *

class BEAR_Event:

    def __init__(self, ip, time, req_type, url, browser, full_agent, l):
        self.ip = ip
        self.time = time
        self.req_type = req_type
        self.url = url
        self.browser = browser
        self.full_agent = full_agent
        self.line = l


def print_set(set_, label, n=10):
    print("PRINTING", label, ", total values", len(set_), ", printing top", n, ":")
    i = 0
    for v in set_:
        print(v)
        i+=1
        if i > n:
            break


def map_events_to_resources(urls):

    ## compile regexp
    compiled_regexp = {}
    for reg in PAPER_MAPPING:
        compiled_regexp[re.compile(reg)] = PAPER_MAPPING[reg]
    matched_urls = set()
    total_urls, total_matched_urls = len(urls), 0

    ## match urls using regexps
    mapper = URLMapper();
    for event in events:
        event_label = mapper.get_event_label(event)
        event.label = event_label
        if event_label:
            total_matched_urls += 1
        matched_urls.add(event_label)
    print('matched', total_matched_urls, 'out of', total_urls)
    return matched_urls


if __name__ == '__main__':

    LOG_PATH = '../../data/bear/findyourhouse_long.log'
    browsers = set()
    urls = set()
    agents = set()
    req_types = set()
    ips = set()

    with open(LOG_PATH) as fr:
        events = []
        for l in fr:
            parts = l.strip().split()
            ip = parts[2]
            time = parts[4].strip('[') + " " + parts[5].strip(']')
            req_type = parts[6].strip("\"")
            url = parts[7]
            browser = parts[12].strip("\"")
            full_agent = " ".join(parts[12:]).strip("\"")
            # for p in parts:
            attrs = [ip, time, req_type, url, browser, full_agent, l]
            for attr in attrs:
                browsers.add(browser)
                urls.add(url)
                agents.add(full_agent)
                req_types.add(req_type)
                ips.add(ip)
            events.append(BEAR_Event(ip, time, req_type, url, browser, full_agent, l))

        resources = map_events_to_resources(events)
        print_set(browsers, "browsers")
        print_set(resources, "resources")
        print_set(agents, "agents")
        print_set(req_types, "req_types")
        print_set(ips, "ips")