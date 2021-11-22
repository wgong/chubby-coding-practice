"""
We want to find employees who badged into our secured room together often. Given an unordered list of names and access times over a single day, find the largest group of people that were in the room together during two or more separate time periods, and the times when they were all present.

badge_records = [
    ["Curtis", "2", "enter"],
    ["John", "1510", "exit"],
    ["John", "455", "enter"],
    ["John", "512", "exit"],
    ["Jennifer", "715", "exit"],
    ["Steve", "815", "enter"],
    ["John", "930", "enter"],
    ["Steve", "1000", "exit"],
    ["Paul", "1", "enter"],
    ["Angela", "1115", "enter"],
    ["Curtis", "1510", "exit"],
    ["Angela", "2045", "exit"],
    ["Nick", "630", "exit"],
    ["Jennifer", "30", "enter"],
    ["Nick", "30", "enter"],
    ["Paul", "2145", "exit"],
]

Expected output:
  Paul, Curtis, and John: 455 to 512, 930 to 1510

For this input data:
  From 455 til 512, the room contains Paul, Curtis and John. Jennifer and Nick are also in the room at this time..
  From 930 til 1510, Paul, Curtis, and John are in the room while Steve and Angela enter and leave, until Curtis leaves at 1510.

The group "Paul, Curtis and John" exists at both of these times, and is the largest group that exists multiple times.

You should note that the group in the expected output is a subset of the people in the room in both cases.

badge_records2 = [
  ["Paul", "1545", "exit"],
  ["Curtis", "1410", "enter"],
  ["Curtis", "222", "enter"],
  ["Curtis", "1630", "exit"],
  ["Paul", "10", "enter"],
  ["Paul", "1410", "enter"],
  ["John", "330", "enter"],
  ["Jennifer", "330", "enter"],
  ["Jennifer", "1410", "exit"],
  ["John", "1410", "exit"],
  ["Curtis", "330", "exit"],
  ["Paul", "330", "exit"],
]

Expected output: 
Curtis, Paul: 222 to 330, 1410 to 1545

"""

"""
candidates = []
Consider all groups (generate subsets)

For each group:
    times_together = []
    room = {}
    latest_entry
    earliest_exit

    1. Order enter/exits (of group) in increasing order of time

    2. Iterate through enter/exits:
        if someone enters:
            latest_entry = enter_time
            add person to room
        else : # exits
            earliest_exit = exit_time

        # `room` should contain people in the room
        if room = group and earliest_exit > latest_entry:
            times_together.append((latest entry, earliest exit))

        if exit:
            remove from room

    3. Compute len(times_together). If this is > 2, add group to candidates[]

Return largest group in candidates
"""

badge_records = [
  ["Curtis", "2", "enter"],
  ["John", "1510", "exit"],
  ["John", "455", "enter"],
  ["John", "512", "exit"],
  ["Jennifer", "715", "exit"],
  ["Steve", "815", "enter"],
  ["John", "930", "enter"],
  ["Steve", "1000", "exit"],
  ["Paul", "1", "enter"],
  ["Angela", "1115", "enter"],
  ["Curtis", "1510", "exit"],
  ["Angela", "2045", "exit"],
  ["Nick", "630", "exit"],
  ["Jennifer", "30", "enter"],
  ["Nick", "30", "enter"],
  ["Paul", "2145", "exit"],
]

def getTimesTogether(group, badge_records):
    times_together = []
    room = set()
    latest_entry = None
    # earliest_exit = None
    
    entry_exit_points = [(int(time), type_, name) for name, time, type_ in badge_records if name in group]
    entry_exit_points.sort()
    
    for time, type_, name in entry_exit_points:
        if type_ == 'enter':
            latest_entry = time
            room.add(name)
        else:
            if room == group:
                times_together.append((latest_entry, time))
            room.remove(name)
    return times_together
    
def largestGroup(badge_records):
    candidate_groups = []
    people = list(set([name for name, _, _ in badge_records]))
    n_people = len(people)
    for n in range(1, 2<<n_people):
        # Generate group
        group = set()
        for i in range(n_people):
            if n & 1<<i:
                group.add(people[i])
        
        times_together = getTimesTogether(group, badge_records)
        if len(times_together) >= 2:
            candidate_groups.append(times_together)

    return max(candidate_groups, key=lambda x: len(x), default=[])

if __name__ == '__main__':
    print(largestGroup(badge_records))