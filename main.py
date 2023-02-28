import asyncio
import websockets
import json
import torch
from agent import Agent
import time
from flask import Flask, render_template, request
import simple_websocket
import numpy as np

app = Flask(__name__)

# create handler for each connection
total_score = 0
record = 0
agent = Agent()
reward = 100
score = 0

@app.route("/", websocket=True)
def echo():
    ws = simple_websocket.Server(request.environ)
    try:
        while True:
            # global variables
            global score
            global reward
            global agent
            global record
            global total_score

            data = ws.receive()

            jsonDataParse = json.loads(data)
            if jsonDataParse["rewardUpdate"] > 0:
                score += 1
            elif jsonDataParse["rewardUpdate"] < 0:
                reward -= abs(jsonDataParse["rewardUpdate"])
                score -= 1
            # get old state
            inputResultsHumanTrainer = []
            raycastsResultsHumanTrainer = []
            OrientationEnemyResultsHumanTrainer = []
            for _ in range(len(jsonDataParse["InputData"])):
                inputResultsHumanTrainer.append(False)
            for _ in range(len(jsonDataParse["raycastsResults"])):
                raycastsResultsHumanTrainer.append(0)
            for _ in range(len(jsonDataParse["EnemyRotation"])):
                OrientationEnemyResultsHumanTrainer.append(0)
            if len(jsonDataParse["raycastsResults"]) > 0:
                for RaycastName, HasCollided in jsonDataParse["raycastsResults"].items():
                    indexForHumanTrainerRaycasts = list(jsonDataParse["raycastsResults"]).index(RaycastName)
                    raycastsResultsHumanTrainer[indexForHumanTrainerRaycasts] = HasCollided
            if len(jsonDataParse["InputData"]) > 0:
                for ActionName, isActive in jsonDataParse["InputData"].items():
                    indexForHumanTrainerInputs = list(jsonDataParse["InputData"]).index(ActionName)
                    inputResultsHumanTrainer[indexForHumanTrainerInputs] = isActive
            if len(jsonDataParse["EnemyRotation"]) > 0:
                for ActionName, isActive in jsonDataParse["EnemyRotation"].items():
                    indexForHumanTrainerOrientationEnemy = list(jsonDataParse["EnemyRotation"]).index(ActionName)
                    OrientationEnemyResultsHumanTrainer[indexForHumanTrainerOrientationEnemy] = isActive
            
            state_old = agent.get_state(
                    [
                        # Raycasts
                        *raycastsResultsHumanTrainer,
                        # Enemy Orientations
                        *OrientationEnemyResultsHumanTrainer,
                        # Distance Enemy
                        jsonDataParse["DistanceEnemy"],
                    ]
            )

            # get move
            final_move = agent.get_action(state_old)
            #print(final_move)
            ws.send(json.dumps(final_move))
    except simple_websocket.ConnectionClosed:
        pass
    return ''


if __name__ == "__main__":
    app.run(host="localhost", port=2943)