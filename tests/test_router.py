from utilis.router_utils import init_router_components, route_answer

def test_router_smoke():
    router, dest_chains, default_chain, retriever = init_router_components()
    out = route_answer("今日は仕事の愚痴を聞いて", router, dest_chains, default_chain, retriever)
    assert isinstance(out, str) and len(out) > 0
