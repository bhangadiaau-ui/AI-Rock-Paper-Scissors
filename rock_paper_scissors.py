import cv2
import mediapipe as mp
import random
import time

# ---------------- SETUP ----------------
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

CHOICES = ["ROCK", "PAPER", "SCISSORS"]

# Scores
p1_score = 0
p2_score = 0

# Game control
MODE = "MENU"     # MENU, AI, MULTI
STATE = "WAIT"    # WAIT → COUNTDOWN → CAPTURE → RESULT
state_start = 0

# Moves
p1_move = None
p2_move = None
result_text = ""

COUNTDOWN_TIME = 3
RESULT_TIME = 2

# ---------------- HELPERS ----------------
def fingers_up(hand):
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    return [hand.landmark[t].y < hand.landmark[p].y for t, p in zip(tips, pips)]

def get_choice(f):
    if f == [False, False, False, False]:
        return "ROCK"
    if f == [True, True, True, True]:
        return "PAPER"
    if f == [True, True, False, False]:
        return "SCISSORS"
    return None

def get_winner(a, b):
    if a == b:
        return "DRAW"
    if (a == "ROCK" and b == "SCISSORS") or \
       (a == "SCISSORS" and b == "PAPER") or \
       (a == "PAPER" and b == "ROCK"):
        return "P1"
    return "P2"

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    now = time.time()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    cv2.rectangle(frame, (0, 0), (w, 120), (0, 0, 0), -1)

    # ---------------- MENU ----------------
    if MODE == "MENU":
        cv2.putText(frame, "Rock Paper Scissors",
                    (90, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(frame, "Press A : AI Mode",
                    (120, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        cv2.putText(frame, "Press M : Multiplayer",
                    (120, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('a'):
            MODE = "AI"
            STATE = "WAIT"
        elif key == ord('m'):
            MODE = "MULTI"
            STATE = "WAIT"

        cv2.imshow("RPS", frame)
        continue

    # ---------------- WAIT FOR HAND(S) ----------------
    if STATE == "WAIT":
        if MODE == "AI" and results.multi_hand_landmarks:
            STATE = "COUNTDOWN"
            state_start = now

        elif MODE == "MULTI" and results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
            STATE = "COUNTDOWN"
            state_start = now

    # ---------------- COUNTDOWN ----------------
    elif STATE == "COUNTDOWN":
        remaining = COUNTDOWN_TIME - int(now - state_start)
        if remaining > 0:
            cv2.putText(frame, str(remaining),
                        (w//2 - 30, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        3, (0,0,255), 5)
        else:
            STATE = "CAPTURE"

    # ---------------- CAPTURE MOVES (ONLY HERE) ----------------
    elif STATE == "CAPTURE":
        if results.multi_hand_landmarks:
            hands_list = results.multi_hand_landmarks

            if MODE == "AI":
                hand = hands_list[0]
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                p1_move = get_choice(fingers_up(hand))
                p2_move = random.choice(CHOICES)

            elif MODE == "MULTI" and len(hands_list) == 2:
                mp_draw.draw_landmarks(frame, hands_list[0], mp_hands.HAND_CONNECTIONS)
                mp_draw.draw_landmarks(frame, hands_list[1], mp_hands.HAND_CONNECTIONS)
                p1_move = get_choice(fingers_up(hands_list[0]))
                p2_move = get_choice(fingers_up(hands_list[1]))

            if p1_move and p2_move:
                win = get_winner(p1_move, p2_move)
                if win == "P1":
                    p1_score += 1
                    result_text = "PLAYER 1 WINS!"
                elif win == "P2":
                    p2_score += 1
                    result_text = "PLAYER 2 WINS!"
                else:
                    result_text = "DRAW!"
                STATE = "RESULT"
                state_start = now

    # ---------------- RESULT ----------------
    elif STATE == "RESULT":
        cv2.putText(frame, f"P1: {p1_move}",
                    (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(frame, f"P2: {p2_move}",
                    (250, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        cv2.putText(frame, result_text,
                    (360, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

        if now - state_start > RESULT_TIME:
            p1_move = None
            p2_move = None
            result_text = ""
            STATE = "WAIT"

    # ---------------- UI ----------------
    cv2.putText(frame, f"P1: {p1_score}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255,255,255), 2)

    cv2.putText(frame, f"P2: {p2_score}",
                (150, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255,255,255), 2)

    cv2.putText(frame, "Q to quit",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255,255,255), 2)

    cv2.imshow("RPS", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
