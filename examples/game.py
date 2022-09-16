
from enum import IntEnum
from dataclasses import dataclass, field
from random import shuffle

from story import MarkovChain

class Action(IntEnum):
    BET = 2,
    CLAIM = 3,
    SPLIT = 4,
    ADVANCE = 5

@dataclass
class Card:
    action: Action
    cost: int
    kwargs_: dict = field(default_factory=dict)

@dataclass
class Deck:
    cards: list = field(default_factory=list)
    used: list = field(default_factory=list, init=False)

    def shuffle(self):
        self.cards = shuffle(self.cards + self.used)
        self.used = []

@dataclass
class Player:
    name: str
    money: int = 0
    score: int = 0
    bets: dict[int, int] = field(default_factory=dict)
    hand: list[Card] = field(default_factory=list, init=False)
    has_claimed: bool = False

class Game:
    def __init__(self, dimension: int = 3, seed: int = 0, players: int = 1):
        self.dimension = dimension

        MarkovChain.reset_static_rng(seed)
        self.chain = MarkovChain.random(dimension, my_seed=seed, max_steps=100)
        self.players = [Player(f'Player #{i}') for i in range(players)]
        self.deck = Deck([
            Card(Action.BET, cost = 0, kwargs_={"bets_num": 2}),
            Card(Action.CLAIM, cost = 0)])
        self.deck.shuffle()

    def give_cards_to(self, player: Player, quantity: int = 1) -> None:
        cards = self.deck.cards[0:quantity]
        player.hand.append(*cards)
    
    def try_play_card(self, player: Player, card_index: int) -> bool:
        if card_index < 0 or card_index >= len(player.hand):
            return False

        card = player.hand[card_index]        
        if card.cost > player.money:
            return False

        player.money -= card.cost

    def play_bet(self, player: Player, bets_num: int):
        choice: int = self.get_input(prompt = f'{player.name}, place your {bets_num} bets!', 
                                     r = [0, self.dimension],
                                     num = bets_num)

        for state in choice:
            if state in player.bets.keys():
                player.bets[state] += 1
            else:
                player.bets[state] = 1

    def execute_claims(self, player: Player):
        if player.has_claimed:
            if self.chain.state in player.bets:
                reward = player.bets[self.chain.state]
                player.score += reward
                del player.bets[self.chain.state]
                player.has_claimed = False
                self.output(f'{player.name}, you earned {reward} score points!')
            else:
                self.output(f'{player.name}, you get no reward this round.')

    def get_input(self, prompt: str, r: tuple[int, int], num: int = 1) -> list | int:
        def verify(choice) -> int:
            x = int(choice)
            if x < r[0] or x >= r[1]:
                raise ValueError
            return x
        print(prompt)
        result = []
        choice: int
        exit_loop = False
        while not exit_loop:
            try:
                choices = input('> ').split(" ")
                choices = map(verify, choices)
                result.extend(choices)
            except ValueError:
                print(f'Your choices should be integer in range {r[0]} to {r[1]}')
            finally:
                if len(result) >= num:
                    exit_loop = True
        if num == 1:
            return result[0]
        return result

    def output(self, msg: str):
        print(msg)

    def run(self):
        for state in self.chain:
            for p in self.players: self.execute_claims(p)
            self.output(f'State is now {state}.')
            for p in self.players:
                action = self.get_input(f'{p.name}, what do you do?', r = (2, 4))
                if action == Action.BET:
                    self.play_bet(p, 2)
                elif action == Action.CLAIM:
                    p.has_claimed = True
            print(self.players[0])
            
if __name__ == "__main__":
    game = Game()
    game.run()