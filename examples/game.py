
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
    kwargs_: dict = field(default_factory=dict)

@dataclass
class Deck:
    cards: list = field(default_factory=list)
    used: list = field(default_factory=list, init=False)
    size: int = field(init=False)

    def __post_init__(self):
        self.size = len(self.cards)

    def reshuffle(self):
        self.cards = self.cards + self.used
        self.used = []
        shuffle(self.cards)

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
            Card(Action.BET, kwargs_={"bets_num": 2}),
            Card(Action.CLAIM,)]*3)
        self.deck.reshuffle()

    def give_cards_to(self, player: Player, num: int = 1) -> None:
        cards = self.deck.cards[0: num]
        player.hand.extend(cards)
        del self.deck.cards[0: num]
    
    def play_card(self, player: Player):
        if len(player.hand) > 0:
            self.output(msg = f'{player.name}, these are your cards:\n{player.hand}')
            choice: int = self.get_input(prompt = f'Which one do you wish to play?', 
                                        r = (0, len(player.hand)))
            card = player.hand[choice]   
            match card.action:
                case Action.BET:
                    self.play_bet(player, **card.kwargs_)
                case Action.CLAIM:
                    self.play_claim(player)
            self.deck.used.append(player.hand.pop(choice))
        else:
            self.output(f'{player.name}, you have no cards.')

    def play_bet(self, player: Player, bets_num: int):
        choice: int = self.get_input(prompt = f'{player.name}, place your {bets_num} bets!', 
                                     r = (0, self.dimension),
                                     num = bets_num)

        for state in choice:
            if state in player.bets.keys():
                player.bets[state] += 1
            else:
                player.bets[state] = 1

    def play_claim(self, player: Player):
        player.has_claimed = True
    
    def execute_claims(self, player: Player):
        if player.has_claimed:
            if self.chain.state in player.bets:
                reward = player.bets[self.chain.state]
                player.score += reward
                del player.bets[self.chain.state]
                self.output(f'{player.name}, you earned {reward} score points for betting on state {self.chain.state}!')
            else:
                self.output(f'{player.name}, you get no reward this round.')
            
            player.has_claimed = False

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
                choices = input('> ').strip().split(" ")
                choices = [verify(choice) for choice in choices]
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
        for p in self.players: self.give_cards_to(p, 2)

        history = []
        for state in self.chain:
            history.append(state)
            print(state, self.chain.state)
            for p in self.players: self.execute_claims(p)

            self.output(f'This is the tape from your machine:\n{history}')
            for p in self.players: self.output(f'{p.name} has {p.score} points and bets as such {p.bets}')

            for p in self.players: self.play_card(p)
            
            if len(self.deck.used) == self.deck.size:
                self.deck.reshuffle()
                for p in self.players: self.give_cards_to(p, 2)
            else:
                for p in self.players: self.give_cards_to(p, 1)
            
            # print(self.players[0])
        for p in self.players: self.output(f'{p.name} has {p.score} points.')

            
if __name__ == "__main__":
    game = Game()
    game.run()