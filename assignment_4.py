from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

import argparse
import sys
import csv
import logging
from time import sleep

from typing import List, Tuple
import uuid
import functools
import ray

from jaeger_client import Config
ray.init(include_dashboard=False)

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


# Note (john): Make sure you use Python's logger to log
#              information about your program
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

# Generates unique operator IDs
def _generate_uuid():
    return uuid.uuid4()


# Custom tuple class with optional metadata
class ATuple:
    """Custom tuple.

    Attributes:
        tuple (Tuple): The actual tuple.
        metadata (string): The tuple metadata (e.g. provenance annotations).
        operator (Operator): A handle to the operator that produced the tuple.
    """
    def __init__(self, tuple, metadata=None, operator=None):
        self.tuple = tuple
        self.metadata = metadata
        self.operator = operator


    # Returns the lineage of self
    def lineage() -> List[ATuple]:
        return self.operator.lineage(self.tuple)
        pass


    # Returns the Where-provenance of the attribute at index 'att_index' of self
    def where(att_index) -> List[Tuple]:
        return self.operator.where(att_index, self.tuple)
        pass

    # Returns the How-provenance of self
    def how() -> string:
        pro = self.operator._metadata
        func_name = pro[-1]
        mid = self.tuple[0][0]
        how = []
        for i in range(len(pro) - 1):
            if(pro[i][4] == mid):
                how.append('f{}*r{}@{}'.format(pro[i][2], pro[i][6], pro[i][5]))
        how_str = str(how).replace("'f", "(f").replace("'", ")").replace("[", "(").replace("]", ")")
        how_str =  func_name + how_str
        return how_str
        pass

    # Returns the input tuples with responsibility \rho >= 0.5 (if any)
    def responsible_inputs() -> List[Tuple]:
        pass


    def __repr__(self):
        return f'ATuple{self.tuple}'

# Data operator
class Operator:
    """Data operator (parent class).

    Attributes:
        id (string): Unique operator ID.
        name (string): Operator name.
        track_prov (bool): Defines whether to keep input-to-output
        mappings (True) or not (False).
        propagate_prov (bool): Defines whether to propagate provenance
        annotations (True) or not (False).
    """
    def __init__(self, id=None, name=None, track_prov=False,
                                           propagate_prov=False, operator = None):
        self.id = _generate_uuid() if id is None else id
        self.name = "Undefined" if name is None else name
        self.track_prov = track_prov
        self.propagate_prov = propagate_prov
        self.operator = operator
        logger.debug("Created {} operator with id {}".format(self.name,
                                                             self.id))

    # NOTE (john): Must be implemented by the subclasses
    def get_next(self):
        logger.error("Method not implemented!")

    # NOTE (john): Must be implemented by the subclasses
    def lineage(self, tuples: List[ATuple]) -> List[List[ATuple]]:
        logger.error("Lineage method not implemented!")

    # NOTE (john): Must be implemented by the subclasses
    def where(self, att_index: int, tuples: List[ATuple]) -> List[List[Tuple]]:
        logger.error("Where-provenance method not implemented!")

# Scan operator
@ray.remote
class Scan(Operator):
    """Scan operator.

    Attributes:
        filepath (string): The path to the input file.
        filter (function): An optional user-defined filter.
        track_prov (bool): Defines whether to keep input-to-output
        mappings (True) or not (False).
        propagate_prov (bool): Defines whether to propagate provenance
        annotations (True) or not (False).
    """
    # Initializes scan operator
    def __init__(self, filepath, filter=None, track_prov=False,
                                              propagate_prov=False, operator=None):
        super().__init__(name="Scan", track_prov=track_prov,
                                   propagate_prov=propagate_prov, operator=operator)
        self.Current = 0
        self.Data = []
        self.filepath = filepath
        self.filter = filter
        self.operator = operator

    # Returns next batch of tuples in given file (or None if file exhausted)
    def get_next(self):
        batchSize = 600
        if self.Current >= len(self.Data):
            return None
        else:
            #if current is less than data size, add batch size here
            next = self.Data[self.Current:self.Current+batchSize]
            self.Current += batchSize
            return tuple(next)

    def execute(self, tuples):
            # span.set_tag(self.name, '100')
            content = []
            #load file and store data as Atuples
            with open(self.filepath, newline='') as f:
                reader = csv.reader(f, delimiter=' ')
                next(reader)
                for line in reader:
                    #if filter is not overridden
                    if self.filter is None:
                        content.append(ATuple(tuple=tuple(line)))
                    else:
                        #append Atuple
                        if self.filter(line):
                            content.append(ATuple(tuple=tuple(line)))
            self.Data = content

            id = self.operator.execute.remote(self.Data, 'span')
            return id

    # Returns the lineage of the given tuples
    def lineage(self, tuples):
        self._lineage = []
        table = {}
        for tuple in tuples:
            table[tuple] = []

        for data in self.Data:
            for tuple in tuples:
                if (data.tuple[1] == str(tuple)):
                    table[tuple].append(data.tuple)

        for key in table:
            self._lineage.append(table[key])
        return self._lineage
        pass

    # Returns the where-provenance of the attribute
    # at index 'att_index' for each tuple in 'tuples'
    def where(self, att_index, tuples):
        self._where = []
        table = {}
        for tuple in tuples:
            table[tuple[att_index]] = []

        line_num = 2
        for data in self.Data:
            for tuple in tuples:
                if (str(tuple[att_index]) == data.tuple[1]):
                    table[tuple[att_index]].append((self.filepath, line_num, data.tuple, data.tuple[2]))
            line_num += 1


        for key in table:
            self._where.append(table[key])
        return self._where
        pass

# Equi-join operator
@ray.remote
class Join(Operator):
    """Equi-join operator.

    Attributes:
        left_input (Operator): A handle to the left input.
        right_input (Operator): A handle to the left input.
        left_join_attribute (int): The index of the left join attribute.
        right_join_attribute (int): The index of the right join attribute.
        track_prov (bool): Defines whether to keep input-to-output
        mappings (True) or not (False).
        propagate_prov (bool): Defines whether to propagate provenance
        annotations (True) or not (False).
    """
    # Initializes join operator
    def __init__(self, left_join_attribute, right_join_attribute,
                                                left_input = None, right_input = None,
                                                track_prov=False,
                                                propagate_prov=False,
                                                operator=None):
        super().__init__(name="Join", track_prov=track_prov,
                                   propagate_prov=propagate_prov,operator=operator)

        self.Data = []
        self.leftData, self.rightData = [], []
        self.left_join_attribute = left_join_attribute
        self.right_join_attribute = right_join_attribute
        self.operator = operator

    # Returns next batch of joined tuples (or None if done)
    def get_next(self):
        batchSize = 600
        if self.Current >= len(self.Data):
            return None
        else:
            next = self.Data[self.Current:self.Current+batchSize]
            self.Current += batchSize
            return tuple(next)

    def execute(self, tuples, pos, sp):
        # with tracer.start_span(self.name, child_of=sp) as span:
            # span.set_tag(self.name, '100')
            if(self.leftData == [] and pos == 'left'):
                self.leftData = tuples
            if(self.rightData == [] and pos == 'right'):
                self.rightData = tuples

            if(self.rightData != [] and self.leftData != []):
                #join left batch and right batch
                for m in self.leftData:
                    for n in self.rightData:

                        if m.tuple[self.left_join_attribute] == n.tuple[self.right_join_attribute]:
                            self.Data.append(ATuple(m.tuple + n.tuple))
                if(self.operator != None):
                    id = self.operator.execute.remote(self.Data, 'span')
                    return id

    # Returns the lineage of the given tuples
    def lineage(self, tuples):
        self._lineage = []
        table = {}
        for tuple in tuples:
            table[tuple] = []

        for data in self.Data:
            for tuple in tuples:
                if (data.tuple[3] == str(tuple)):
                    table[tuple].append(data.tuple[:2])
                    table[tuple].append(data.tuple[2:])

        for key in table:
            self._lineage.append(table[key])
        return self._lineage
        pass

    # Returns the where-provenance of the attribute
    # at index 'att_index' for each tuple in 'tuples'
    def where(self, att_index, tuples):
        return self._track.where(att_index, tuples)
        pass

# Project operator
@ray.remote
class Project(Operator):
    """Project operator.

    Attributes:
        input (Operator): A handle to the input.
        fields_to_keep (List(int)): A list of attribute indices to keep.
        If empty, the project operator behaves like an identity map, i.e., it
        produces and output that is identical to its input.
        track_prov (bool): Defines whether to keep input-to-output
        mappings (True) or not (False).
        propagate_prov (bool): Defines whether to propagate provenance
        annotations (True) or not (False).
    """
    # Initializes project operator
    def __init__(self, input = None, fields_to_keep=[], track_prov=False,
                                                 propagate_prov=False, operator=None):
        super().__init__(name="Project", track_prov=track_prov,
                                      propagate_prov=propagate_prov, operator=operator)
        self.Current = 0
        self.Data = []
        self.fields_to_keep = fields_to_keep
        self.operator = operator


    # Return next batch of projected tuples (or None if done)

    def get_data(self):
        return self.Data

    def execute(self, tuples, sp):
        # with tracer.start_span(self.name, child_of=sp) as span:
            # span.set_tag(self.name, '100')
            for i in tuples:
                newData = []
                #extract data
                for content in self.fields_to_keep:
                    newData.append(i.tuple[content])
                self.Data.append(ATuple(tuple(newData)))

            if(self.operator is not None):
                id = self.operator.remote(self.Data, 'span')
                return id


    # Returns the lineage of the given tuples
    def lineage(self, tuples):
        return self._track.lineage(tuples)
        pass

    # Returns the where-provenance of the attribute
    # at index 'att_index' for each tuple in 'tuples'
    def where(self, att_index, tuples):
        return self._track.where(att_index, tuples)
        pass

# Group-by operator
@ray.remote
class GroupBy(Operator):
    """Group-by operator.

    Attributes:
        input (Operator): A handle to the input
        key (int): The index of the key to group tuples.
        value (int): The index of the attribute we want to aggregate.
        agg_fun (function): The aggregation function (e.g. Average)
        track_prov (bool): Defines whether to keep input-to-output
        mappings (True) or not (False).
        propagate_prov (bool): Defines whether to propagate provenance
        annotations (True) or not (False).
    """
    # Initializes average operator
    def __init__(self, key, value, agg_gun, track_prov=False,
                                                   propagate_prov=False, operator=None):
        super().__init__(name="GroupBy", track_prov=track_prov,
                                      propagate_prov=propagate_prov, operator=operator)

        self.Current = 0
        self.Data = []
        self.operator = operator
        self.key = key
        self.value = value
        self.agg_gun = agg_gun

    # Returns aggregated value per distinct key in the input (or None if done)
    def get_next(self):
        batchSize = 600
        if self.Current >= len(self.Data):
            return None
        else:
            next = self.Data[self.Current:self.Current+batchSize]
            self.Current += batchSize
            return tuple(next)

    # Returns the lineage of the given tuples
    def lineage(self, tuples):
        return self._track.lineage(tuples)
        pass

    # Returns the where-provenance of the attribute
    # at index 'att_index' for each tuple in 'tuples'
    def where(self, att_index, tuples):
        return self._track.where(att_index, tuples)
        pass

    def execute(self, tuples, sp):
        # with tracer.start_span(self.name, child_of=sp) as span:
            # span.set_tag(self.name, '100')
            groupDict = {}
            for i in tuples:
                #if the key of this tuple has existed
                if i.tuple[self.key] in groupDict:
                    groupDict[i.tuple[self.key]].append(ATuple(tuple(i.tuple[self.value])))

                else:
                    groupDict[i.tuple[self.key]] = [ATuple(tuple(i.tuple[self.value]))]

            #use aggregation
            for i, j in groupDict.items():
                # AGG = ray.get(Average.remote(j))[0]
                AGG = j[0].tuple[0]
                self.Data.append(ATuple((i, AGG)))

            if(self.operator is not None):
                self.operator.execute.remote(self.Data, 'span')


# Custom histogram operator
@ray.remote
class Histogram(Operator):
    """Histogram operator.

    Attributes:
        input (Operator): A handle to the input
        key (int): The index of the key to group tuples. The operator outputs
        the total number of tuples per distinct key.
        track_prov (bool): Defines whether to keep input-to-output
        mappings (True) or not (False).
        propagate_prov (bool): Defines whether to propagate provenance
        annotations (True) or not (False).
    """
    # Initializes histogram operator
    def __init__(self, input, key=0, track_prov=False, propagate_prov=False, operator=None):
        super().__init__(name="Histogram",
                                        track_prov=track_prov,
                                        propagate_prov=propagate_prov, operator=operator)
        self.Current = 0
        self.Data = []

        histDict = {}
        cur = ray.get(input.get_next.remote())

        while cur:
            for i in cur:
                #count the tuple with key in histogram dictionary
                if i.tuple[key] in histDict:
                    histDict[i.tuple[key]] += 1
                else:
                    histDict[i.tuple[key]] = 0

            cur = ray.get(input.get_next.remote())
        for i, j in histDict.items():
            self.Data.append(ATuple((i, j+1)))
        #sort values
        self.Data.sort(key=lambda x: int(x.tuple[0]))

    # Returns histogram (or None if done)
    def get_next(self):
        batchSize = 600
        if self.Current >= len(self.Data):
            return None
        else:
            next = self.Data[self.Current:self.Current+batchSize]
            self.Current += batchSize
            return tuple(next)

# Order by operator
@ray.remote
class OrderBy(Operator):
    """OrderBy operator.

    Attributes:
        input (Operator): A handle to the input
        comparator (function): The user-defined comparator used for sorting the
        input tuples.
        ASC (bool): True if sorting in ascending order, False otherwise.
        track_prov (bool): Defines whether to keep input-to-output
        mappings (True) or not (False).
        propagate_prov (bool): Defines whether to propagate provenance
        annotations (True) or not (False).
    """
    # Initializes order-by operator
    def __init__(self, comparator, ASC=True, track_prov=False,
                                                    propagate_prov=False, operator=None):
        super().__init__(name="OrderBy",
                                      track_prov=track_prov,
                                      propagate_prov=propagate_prov,operator=operator)
        self.Current = 0
        self.Data = []
        self.comparator = comparator
        self.ASC = ASC
        self.operator = operator

    def get_next(self):
        batchSize = 600
        if self.Current >= len(self.Data):
            return None
        else:
            next = self.Data[self.Current:self.Current+batchSize]
            self.Current += batchSize
            return tuple(next)

    def execute(self, tuples, sp):
        # with tracer.start_span(self.name, child_of=sp) as span:
            # span.set_tag(self.name, '100')
            self.Data.extend(tuples)
            #use conparator to sort in ASC order
            self.Data = sorted(self.Data, key=functools.cmp_to_key(self.comparator), reverse=not self.ASC)
            if(self.operator is not None):
                self.operator.execute.remote(self.Data, 'span')


    # Returns the lineage of the given tuples
    def lineage(self, tuples):
        return self._track.lineage(tuples)
        pass

    # Returns the where-provenance of the attribute
    # at index 'att_index' for each tuple in 'tuples'
    def where(self, att_index, tuples):
        return self._track.where(att_index, tuples)
        pass

# Limit operator
@ray.remote
class Limit(Operator):
    """Limit operator.

    Attributes:
        input (Operator): A handle to the input.
        n (int): The maximum number of tuples to output.
        track_prov (bool): Defines whether to keep input-to-output
        mappings (True) or not (False).
        propagate_prov (bool): Defines whether to propagate provenance
        annotations (True) or not (False).
    """
    # Initializes top-k operator
    def __init__(self, n, track_prov=False, propagate_prov=False, operator=None):
        super().__init__(name="Limit", track_prov=track_prov,
                                   propagate_prov=propagate_prov, operator=operator)
        self.Current = 0
        self.Data = []
        self.n = n
        self.operator = operator


    # Returns the first k tuples in the input (or None if done)
    def get_next(self):
        batchSize = 600
        if self.Current >= len(self.Data):
            return None
        else:
            next = self.Data[self.Current:self.Current+batchSize]
            self.Current += batchSize
            return tuple(next)

    def execute(self, tuples, sp):
        # with tracer.start_span(self.name, child_of=sp) as span:
            # span.set_tag(self.name, '100')
            while self.n:
                for i in tuples:
                    self.Data.append(i)
                    self.n  = self.n - 1
                    if self.n == 0:
                        break
            if(self.operator is not None):
                self.operator.execute.remote(self.Data, 'span')

@ray.remote
def Average(input):
    inputs = input
    count = 0
    content = []
    for i in inputs:
        if content:
            for j in range(len(i.tuple)):
                content[j] += int(i.tuple[j])
        else:
            content = [int(k) for k in i.tuple]
        count += 1
    return [i / count for i in content]


# Top-k operator
@ray.remote
class TopK(Operator):
    """TopK operator.

    Attributes:
        input (Operator): A handle to the input.
        k (int): The maximum number of tuples to output.
        track_prov (bool): Defines whether to keep input-to-output
        mappings (True) or not (False).
        propagate_prov (bool): Defines whether to propagate provenance
        annotations (True) or not (False).
    """
    # Initializes top-k operator
    def __init__(self, input, k=None, track_prov=False, propagate_prov=False, operator=None):
        super().__init__(name="TopK", track_prov=track_prov,
                                   propagate_prov=propagate_prov, operator=operator)
        self.Current = 0
        self.Data = []

        cur = ray.get(input.get_next.remote())

        while cur:
            cur = sortedInput.get_next()

    # Returns the first k tuples in the input (or None if done)
    def get_next(self):
        batchSize = 600
        if self.Current >= len(self.Data):
            return None
        else:
            next = self.Data[self.Current:self.Current+batchSize]
            self.Current += batchSize
            return tuple(next)

    # Returns the lineage of the given tuples
    def lineage(self, tuples):
        return self._track.lineage(tuples)
        pass

    # Returns the where-provenance of the attribute
    # at index 'att_index' for each tuple in 'tuples'
    def where(self, att_index, tuples):
        return self._track.where(att_index, tuples)
        pass

# Select operator
@ray.remote
class Select(Operator):
    """Select operator.

    Attributes:
        input (Operator): A handle to the input.
        predicate (function): The selection predicate.
        track_prov (bool): Defines whether to keep input-to-output
        mappings (True) or not (False).
        propagate_prov (bool): Defines whether to propagate provenance
        annotations (True) or not (False).
    """
    # Initializes select operator
    def __init__(self, pos, predicate, input = None, track_prov=False,
                                         propagate_prov=False,operator=None):
        super().__init__(name="Select", track_prov=track_prov,
                                     propagate_prov=propagate_prov, operator=operator)
        self.Data = []
        self.predicate = predicate
        self.pos = pos
        self.operator = operator

    # Returns next batch of tuples that pass the filter (or None if done)
    def get_next(self):
        batchSize = 600
        if self.Current >= len(self.Data):
            return None
        else:
            next = self.Data[self.Current:self.Current+batchSize]
            self.Current += batchSize
            return tuple(next)

    def execute(self, tuples, sp):
        # with tracer.start_span(self.name, child_of=sp) as span:
            # span.set_tag(self.name, '100')
            for i in tuples:
                #check if tuple is what we want to select
                if self.predicate(i.tuple):
                    self.Data.append(i)

            if(self.operator is not None):
                id = self.operator.execute.remote(self.Data, self.pos, 'span')
                return id

# Sink Operator
@ray.remote
class Sink(Operator):
    def __init__(self, track_prov=False, propagate_prov=False, operator=None):
        super().__init__(name='Sink', track_prov=track_prov, propagate_prov=propagate_prov, operator=operator)
        self.Data = []

    def get_result(self, id):
        # with tracer.start_span(self.name) as span:
            # span.set_tag(self.name, '100')
            return ray.get(id.get_data.remote())


if __name__ == "__main__":
    logger.info("Assignment #1")
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=int, required=True)
    parser.add_argument('--friends', type=str, required=True)
    parser.add_argument('--ratings', type=str, required=True)
    parser.add_argument('--uid', type=int, required=True)
    #in task 2, argument movie id is not fundamental
    parser.add_argument('--mid', type=int)
    args = parser.parse_args()
    print('Arguments:', args)


    if args.task == 4:

        project = Project.remote(fields_to_keep = [0])
        Limits = Limit.remote(n = 1, operator = project)
        OrderBys = OrderBy.remote(comparator = lambda x1, x2: int(x1.tuple[1]) - int(x2.tuple[1]), ASC=False, operator = Limits)
        GroupBys = GroupBy.remote(key = 3, value = 4, agg_gun = Average, operator = OrderBys)
        Joins = Join.remote(left_join_attribute = 1,right_join_attribute = 0, operator = GroupBys)

        FriendsUid = Select.remote(predicate = lambda x: int(x[0]) == args.uid, operator = Joins, pos = 'left')
        RatingsMid = Select.remote(predicate = lambda x: int(x[1]) == args.mid, operator = Joins, pos = 'right')

        Friends = Scan.remote(args.friends, operator = FriendsUid)
        Ratings = Scan.remote(args.ratings, operator = RatingsMid)
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
        print(f'The recommendation query for User {args.uid} is {Recommend}')
