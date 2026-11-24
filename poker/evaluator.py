from itertools import combinations
from .card import Card

def get_hand_rank(hand): # Evaluates The Rank of a 5 card hand
    ranks = sorted([card.rank_value for card in hand], reverse=True) # Sorts cards by rank value
    suits = [card.suit for card in hand]  # Creates list of suits in hand

    is_flush = len(set(suits)) == 1 # Checks if all cards are suited 
    is_straight = (max(ranks) - min(ranks) == 4) and len(set(ranks)) == 5 # Checks if cards are consecutive uses the indices

    if set(ranks) == {12, 0, 1, 2, 3}:  # Special case for A-2-3-4-5 straight
        is_straight = True
        ranks = [3, 2, 1, 0, -1]  # Treated Ace as low
    
    rank_counts = {} # Counts of each rank in hand
    for r in ranks:
        rank_counts[r] = rank_counts.get(r, 0) + 1

    ranks = sorted(ranks, key=lambda r: (rank_counts[r], r), reverse=True) # Pairs/trips/quads first, then kickers, so KK543 beats 22AKQ
    counts = sorted(rank_counts.values(), reverse=True) # Sorts counts to determine pairs, trips, quads

    if is_straight and is_flush:
        return (8, ranks)  # Straight flush
    if counts == [4, 1]:
        return (7, ranks)  # Four of a kind
    if counts == [3, 2]:
        return (6, ranks)  # Full house
    if is_flush:
        return (5, ranks)  # Flush
    if is_straight:
        return (4, ranks)  # Straight
    if counts == [3, 1, 1]:
        return (3, ranks)  # Three of a kind
    if counts == [2, 2, 1]:
        return (2, ranks)  # Two pair
    if counts == [2, 1, 1, 1]:
        return (1, ranks)  # One pair
    return (0, ranks)  # High card

def best_hand(cards): # Checks which combo of 5 cards from 7 has the best rank
    best = None
    for combo in combinations(cards, 5): # Generates all possible 5 card combos from the 7 cards
        rank = get_hand_rank(combo)
        if best is None or rank > best:
            best = rank # loops through all combos and keeps track of the best one found so far
    return best



# --- fast path: one number per hand, higher is better. Uses the compiled phevaluator library when it's
# installed (about 50x faster); otherwise falls back to best_hand above, which gives the same order.
try:
    from phevaluator import evaluate_cards as _fast
except ImportError:
    _fast = None

SUIT_ORDER = {"Clubs": 0, "Diamonds": 1, "Hearts": 2, "Spades": 3}  # phevaluator numbers cards rank * 4 + suit
_ids = {}
# phevaluator ranks 1 (royal flush) to 7462 (worst high card); these are where each category starts
_CATEGORY_STARTS = [(10, 8), (166, 7), (322, 6), (1599, 5), (1609, 4), (2467, 3), (3325, 2), (6185, 1), (7462, 0)]


def card_id(card): # Our Card -> phevaluator's card number
    key = (card.rank, card.suit)
    if key not in _ids:
        _ids[key] = Card.RANKS.index(card.rank) * 4 + SUIT_ORDER[card.suit]
    return _ids[key]


def hand_value(cards): # Comparable strength of the best 5-card hand in these cards: higher wins
    if _fast is None:
        return best_hand(cards)
    return 7463 - _fast(*[card_id(c) for c in cards])


def value_of_ids(ids): # Same, from card numbers directly (fastest; needs phevaluator)
    return 7463 - _fast(*ids)


def category(value): # 0 = high card ... 8 = straight flush
    if not isinstance(value, int):
        return value[0]
    rank = 7463 - value
    return next(cat for top, cat in _CATEGORY_STARTS if rank <= top)
