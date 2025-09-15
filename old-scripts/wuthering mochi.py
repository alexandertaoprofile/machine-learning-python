import random
import numpy as np
import pandas as pd

# Define character skills
SKILLS = {
    '收束的未来': lambda player: player.adjust_forced_roll([2, 3]),
    '如影随形': lambda player: player.extra_if_last(offset=3),
    '令尹之名': lambda player: player.top_if_stack(prob=0.4),
    '谋而后定': lambda player: player.last_next_turn(prob=0.65),
    '一个人的狂欢': lambda player: player.once_solo_celebration(prob=0.5),
    '利润加倍': lambda player: player.double_move(prob=0.28),
    '加班在左，调休在右': lambda player: player.roll_one_three_only([1, 3], prob_extra=0.4),
    '岁主庇佑': lambda player: player.extra_move(prob=0.5, extra=1),
    '先声夺人': lambda player: player.first_extra(offset=2),
    '寄生的蔓足': lambda player: player.parasitic_stack(),
    '大轴登场': lambda player: player.last_extra(offset=2),
    '翻盘桥段': lambda player: player.comeback_buff(prob=0.6, extra=2)
}

class Player:
    def __init__(self, name, skill):
        self.name = name
        self.skill = skill
        self.reset()

    def reset(self):
        self.position = 0
        self.stack = []
        self.extra_moves = 0
        self.forced_roll = None
        self.last_roll = 0
        self.next_forced_last = False
        self.solo_active = False
        self.comeback_active = False
        self.used_parasitic = False

    # Skill implementations
    def adjust_forced_roll(self, allowed):
        self.forced_roll = allowed

    def extra_if_last(self, offset):
        if self.is_last():
            self.extra_moves += offset

    def top_if_stack(self, prob):
        if self.stack and random.random() < prob:
            # move to top of its stack
            self.stack.insert(0, self.stack.pop())

    def last_next_turn(self, prob):
        if random.random() < prob and self.stack:
            self.next_forced_last = True

    def once_solo_celebration(self, prob):
        if random.random() < prob:
            self.solo_active = True

    def double_move(self, prob):
        if random.random() < prob:
            self.extra_moves += self.last_roll * 2

    def roll_one_three_only(self, allowed, prob_extra):
        self.forced_roll = allowed
        if self.stack and random.random() < prob_extra:
            self.extra_moves += 2

    def extra_move(self, prob, extra):
        if random.random() < prob:
            self.extra_moves += extra

    def first_extra(self, offset):
        if self.simulator.move_counter == 0:
            self.extra_moves += offset

    def parasitic_stack(self):
        if not self.used_parasitic:
            self.simulator.parasitic_flag = True
            self.used_parasitic = True

    def last_extra(self, offset):
        if self.simulator.current_player_index == len(self.simulator.characters) - 1:
            self.extra_moves += offset

    def comeback_buff(self, prob, extra):
        if not self.comeback_active and self.is_last() and random.random() < prob:
            self.comeback_active = True
            self.extra_moves += extra

    def is_last(self):
        # determine if currently last by position
        positions = [p.position for p in self.simulator.characters]
        return self.position == min(positions)

class GameSimulator:
    def __init__(self, characters, track_length):
        self.characters = characters
        self.track_length = track_length
        for p in self.characters:
            p.simulator = self
        self.move_counter = 0
        self.parasitic_flag = False

    def simulate_one(self):
        # reset players
        for p in self.characters:
            p.reset()
        self.move_counter = 0
        self.force_last_queue = []
        self.parasitic_flag = False

        order = list(self.characters)
        rankings = []

        while order:
            for player in list(order):
                self.current_player_index = order.index(player)

                # force-last logic
                if player.next_forced_last:
                    continue

                # roll dice
                if player.forced_roll:
                    roll = random.choice(player.forced_roll)
                else:
                    roll = random.randint(1, 6)
                player.last_roll = roll

                # apply skill triggers
                SKILLS[player.skill](player)

                # calculate movement
                steps = roll + player.extra_moves
                player.position += steps
                player.extra_moves = 0

                # parasitic stacking
                if self.parasitic_flag:
                    idx = order.index(player)
                    if idx < len(order) - 1:
                        player.stack.extend(order[idx+1:])
                    self.parasitic_flag = False

                # finish line check
                if player.position >= self.track_length:
                    rankings.append(player)
                    order.remove(player)
                    continue

                # queue next forced-last
                if player.next_forced_last:
                    self.force_last_queue.append(player)
                    player.next_forced_last = False

                self.move_counter += 1

            # process forced-last at end of round
            for player in list(self.force_last_queue):
                if player not in order:
                    continue
                roll = random.randint(1, 6)
                player.last_roll = roll
                SKILLS[player.skill](player)
                player.position += roll + player.extra_moves
                player.extra_moves = 0
                if player.position >= self.track_length:
                    rankings.append(player)
                    order.remove(player)
                player.next_forced_last = False
            self.force_last_queue.clear()

        # return list of names in finish order
        return [p.name for p in rankings]

    def simulate(self, trials=10000):
        win_count = {p.name: 0 for p in self.characters}
        rank_sums = {p.name: 0 for p in self.characters}

        for _ in range(trials):
            results = self.simulate_one()
            for rank, name in enumerate(results, start=1):
                rank_sums[name] += rank
            win_count[results[0]] += 1

        win_rate = {name: count / trials for name, count in win_count.items()}
        avg_rank = {name: rank_sums[name] / trials for name in rank_sums}
        return win_rate, avg_rank

# Example usage
if __name__ == '__main__':
    chars = [Player('A', '如影随形'), Player('B', '先声夺人'), Player('C', '收束的未来')]
    sim = GameSimulator(chars, track_length=50)
    win_rate, avg_rank = sim.simulate(trials=5000)

    print("胜率和平均名次：")
    for name in win_rate:
        print(f"{name}: 胜率 {win_rate[name]*100:.2f}%, 平均名次 {avg_rank[name]:.2f}")
