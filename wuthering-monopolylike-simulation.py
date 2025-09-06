import random
import numpy as np
import pandas as pd

# 提供一个团子列表
available_players = [
    '大轴登场', '先声夺人', '岁主庇佑', '加班在左，调休在右',
    '如影随形', '收束的未来', '令尹之名', '翻盘桥段', '寄生的蔓足', '谋而后定'
]

class Player:
    def __init__(self, name, skill):
        self.name = name
        self.skill = skill
        self.sim = None
        self.reset()

    def reset(self):
        self.position = 0
        self.stack = []
        self.forced_roll = None      # restrict dice for this turn
        self.next_extra = 0          # extra moves next turn
        self.extra = 0               # extra moves this turn
        self.flag_force_last = False # move order for next turn
        self.flag_parasitic = False
        self.flag_solo = False       # solo extra-per-player this turn
        self.flag_double = False
        self.flag_comeback = False

    # Skill support methods
    def set_forced_roll(self, allowed):
        self.forced_roll = allowed.copy()

    def extra_if_last(self, offset):
        if self.is_last():
            self.extra += offset

    def extra_if_first(self, offset):
        if self.sim.move_counter == 0:
            self.extra += offset

    def top_stack_with_prob(self, prob):
        if self.stack and random.random() < prob:
            # move this player to top within its stack group
            self.stack.insert(0, self.stack.pop())

    def flag_force_last_next_turn(self, prob):
        if self.stack and random.random() < prob:
            self.flag_force_last = True

    def flag_solo_steps(self, prob):
        if random.random() < prob:
            self.flag_solo = True

    def flag_double_move(self, prob):
        if random.random() < prob:
            self.flag_double = True

    def prepare_roll_1_or_3(self, prob_extra_next):
        # this turn roll only 1 or 3
        self.forced_roll = [1, 3]
        # schedule extra on next turn if stacked now
        if self.stack and random.random() < prob_extra_next:
            self.next_extra += 2

    def flag_extra_move_current(self, prob, extra):
        if random.random() < prob:
            self.extra += extra

    def flag_parasitic_stack(self):
        if not self.flag_parasitic:
            self.flag_parasitic = True

    def flag_comeback_buff(self, prob, extra):
        if not self.flag_comeback and self.is_last() and random.random() < prob:
            self.flag_comeback = True
            self.next_extra += extra

    def is_last(self):
        pos = [p.position for p in self.sim.characters]
        return self.position == min(pos)


class GameSimulator:
    def __init__(self, characters, track_len, initial_steps=None, initial_stack=None):
        self.characters = characters
        self.track_len = track_len
        self.init_steps = initial_steps or {}
        self.init_stack = initial_stack or {}
        for p in self.characters:
            p.sim = self
        self.move_counter = 0

    def simulate_one(self):
        # reset and initial conditions
        for p in self.characters:
            p.reset()
            if p.name in self.init_steps:
                p.position = self.track_len - self.init_steps[p.name]
        for bot, tops in self.init_stack.items():
            base = next(p for p in self.characters if p.name == bot)
            base.stack = [next(q for q in self.characters if q.name == t) for t in tops]

        order = list(self.characters)
        rankings = []
        # main loop
        while order:
            # one round
            for p in order.copy():
                # check force-last
                if p.flag_force_last:
                    continue
                # roll
                if p.forced_roll:
                    roll = random.choice(p.forced_roll)
                else:
                    roll = random.randint(1, 6)
                # base steps
                steps = roll
                # apply current extras
                if p.flag_double:
                    steps += roll  # double move
                    p.flag_double = False
                if p.flag_solo:
                    steps += len(p.sim.characters) - 1
                    p.flag_solo = False
                steps += p.extra
                p.extra = 0
                # move
                p.position += steps
                # parasitic
                if p.flag_parasitic:
                    idx = order.index(p)
                    p.stack.extend(order[idx + 1:])
                    p.flag_parasitic = False
                # finish
                if p.position >= self.track_len:
                    rankings.append(p)
                    order.remove(p)
                    continue
                # schedule next-turn extras/order
                if p.flag_force_last:
                    # stays for next round
                    pass
                if p.next_extra:
                    p.extra = p.next_extra
                    p.next_extra = 0
                p.flag_force_last = False
                # increment move counter
                self.move_counter += 1
            # handle those flagged force-last
            for p in [q for q in self.characters if q.flag_force_last and q in order]:
                roll = random.randint(1, 6)
                steps = roll + p.extra
                p.extra = 0
                p.position += steps
                if p.position >= self.track_len:
                    rankings.append(p)
                    order.remove(p)
                p.flag_force_last = False
            self.move_counter = 0
        return [p.name for p in rankings]

    def simulate(self, trials=10000):
        win = {p.name: 0 for p in self.characters}
        rank_sum = {p.name: 0 for p in self.characters}
        for _ in range(trials):
            res = self.simulate_one()
            for i, name in enumerate(res, 1): rank_sum[name] += i
            win[res[0]] += 1
        return ({k: v / trials for k, v in win.items()}, {k: v / trials for k, v in rank_sum.items()})


def select_players():
    print("可选团子：")
    for i, player in enumerate(available_players, 1):
        print(f"{i}. {player}")

    selected_indices = []
    while len(selected_indices) < 4:
        try:
            selected = int(input("请输入选择的团子编号（1-10）: "))
            if selected < 1 or selected > len(available_players):
                print("无效编号，请重新选择。")
            elif selected not in selected_indices:
                selected_indices.append(selected)
            else:
                print("已选择该团子，请选择不同的团子。")
        except ValueError:
            print("无效输入，请输入一个数字。")
    
    selected_players = [available_players[i-1] for i in selected_indices]
    return selected_players


def select_order(selected_players):
    print("\n选择比赛顺序：")
    for i, player in enumerate(selected_players, 1):
        print(f"{i}. {player}")
    
    order_indices = []
    while len(order_indices) < len(selected_players):
        try:
            selected = int(input(f"请输入第{len(order_indices)+1}个团子的编号（1-{len(selected_players)}）: "))
            if selected < 1 or selected > len(selected_players):
                print("无效编号，请重新选择。")
            elif selected not in order_indices:
                order_indices.append(selected)
            else:
                print("已选择该团子，请选择不同的团子。")
        except ValueError:
            print("无效输入，请输入一个数字。")
    
    ordered_players = [selected_players[i-1] for i in order_indices]
    return ordered_players


def select_track_length():
    while True:
        try:
            length = int(input("请输入比赛的赛道长度（推荐16或更长的步数，建议至少20步）: "))
            if length <= 0:
                print("赛道长度应大于0，请重新输入。")
            else:
                return length
        except ValueError:
            print("无效输入，请输入一个数字。")


def main():
    # 选择团子
    selected_players = select_players()
    # 选择比赛顺序
    ordered_players = select_order(selected_players)
    # 选择赛道长度
    track_length = select_track_length()

    print("\n选择的团子和比赛顺序：")
    for i, player in enumerate(ordered_players, 1):
        print(f"第{i}个出场：{player}")

    print(f"\n赛道长度为：{track_length}步")

    # 初始化团子并开始比赛模拟
    chars = [Player(player, player) for player in ordered_players]
    initial_steps = {player: 15 + i for i, player in enumerate(ordered_players)}  # 根据顺序决定步数
    sim = GameSimulator(characters=chars, track_len=track_length, initial_steps=initial_steps)
    win_rates, avg_ranks = sim.simulate(trials=5000)

    print("\n=== 模拟结果 (5000 场) ===")
    for name in chars:
        wr = win_rates[name.name] * 100
        ar = avg_ranks[name.name]
        print(f"{name.name}: 胜率 {wr:.2f}%, 平均名次 {ar:.2f}")


if __name__ == "__main__":
    main()
