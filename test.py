import pytest
from assignment_4 import Scan, Select, Project, Average, Join, GroupBy, OrderBy, Limit, Sink
from time import sleep
from typing import List, Tuple
import ray
import logging
from jaeger_client import Config

@pytest.fixture()
def setup():
    return {'friends': '../dataset/friends.txt', 'ratings': '../dataset/movie_ratings.txt', 'uid': 12, 'mid': 480}

def test(setup):
    ray.init(include_dashboard=False, _redis_password='lishuai110')

    def init_tracer(service):
        logging.getLogger('').handlers = []
        logging.basicConfig(format='%(message)s', level=logging.DEBUG)
        config = Config(
            config={
                'sampler': {
                    'type': 'const',
                    'param': 1,
                },
                'logging': True,
            },
            service_name=service,
        )
        # this call also sets opentracing.tracer
        return config.initialize_tracer()

    tracer = init_tracer('recommondation')

    project = Project.remote(fields_to_keep = [0])
    Limits = Limit.remote(n = 1, operator = project)
    OrderBys = OrderBy.remote(comparator = lambda x1, x2: int(x1.tuple[1]) - int(x2.tuple[1]), ASC=False, operator = Limits)
    GroupBys = GroupBy.remote(key = 3, value = 4, agg_gun = Average, operator = OrderBys)
    Joins = Join.remote(left_join_attribute = 1,right_join_attribute = 0, operator = GroupBys)

    FriendsUid = Select.remote(predicate = lambda x: int(x[0]) == setup['uid'], operator = Joins, pos = 'left')
    RatingsMid = Select.remote(predicate = lambda x: int(x[1]) == setup['mid'], operator = Joins, pos = 'right')

    Friends = Scan.remote(setup['friends'], operator = FriendsUid)
    Ratings = Scan.remote(setup['ratings'], operator = RatingsMid)
    sink = Sink.remote()
    sleep(3)
    with tracer.start_span('Scan') as span1:
        span1.set_tag('scan', '80')
        Friends.execute.remote(None)
        with tracer.start_span('Select', child_of=span1) as span2:
            span2.set_tag('select', '80')
            with tracer.start_span('Join', child_of=span2) as span3:
                span3.set_tag('join', '80')
                with tracer.start_span('GroupBy', child_of=span3) as span4:
                    span4.set_tag('groupby', '80')
                    with tracer.start_span('OrderBy', child_of=span4) as span5:
                        span5.set_tag('orderby', '80')
                        with tracer.start_span('Limit', child_of=span5) as span6:
                            span6.set_tag('limit', '80')
                            with tracer.start_span('Project', child_of=span6) as span7:
                                span7.set_tag('project', '80')
                                with tracer.start_span('Sink', child_of=span7) as span8:
                                    span8.set_tag('sink','80')

    with tracer.start_span('Scan') as span1:
        span1.set_tag('scan', '80')
        Ratings.execute.remote(None)
        with tracer.start_span('Select', child_of=span1) as span2:
            span2.set_tag('select', '80')
            with tracer.start_span('Join', child_of=span2) as span3:
                span3.set_tag('join', '80')
                with tracer.start_span('GroupBy', child_of=span3) as span4:
                    span4.set_tag('groupby', '80')
                    with tracer.start_span('OrderBy', child_of=span4) as span5:
                        span5.set_tag('orderby', '80')
                        with tracer.start_span('Limit', child_of=span5) as span6:
                            span6.set_tag('limit', '80')
                            with tracer.start_span('Project', child_of=span6) as span7:
                                span7.set_tag('project', '80')
                                with tracer.start_span('Sink', child_of=span7) as span8:
                                    span8.set_tag('sink', '80')

    sleep(5)
    Recommend = ray.get(sink.get_result.remote(project))[0].tuple[0]
    assert Recommend == '480'
