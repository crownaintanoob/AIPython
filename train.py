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
reward = 0
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
                score = score + 1
            elif jsonDataParse["rewardUpdate"] < 0:
                reward = reward + abs(jsonDataParse["rewardUpdate"])
                score = score - 1
            if jsonDataParse["done"] == False:
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
                #final_move = agent.get_action(state_old)
                final_move = []
                for _ in range(len(jsonDataParse["InputData"])):
                    final_move.append(0)
                for ActionName, isActive in jsonDataParse["InputData"].items():
                    indexHTrainerFinalMove = list(jsonDataParse["InputData"]).index(ActionName)
                    if isActive == False:
                        final_move[indexHTrainerFinalMove] = 0
                    else:
                        final_move[indexHTrainerFinalMove] = 1
                #print(final_move)

                # perform move and get new state
                state_new = agent.get_state(
                    [
                        # Raycasts
                        *raycastsResultsHumanTrainer,
                        # Enemy Orientations
                        *OrientationEnemyResultsHumanTrainer,
                        # Distance Enemy
                        jsonDataParse["DistanceEnemy"],
                    ]
                )

                # train short memory
                agent.train_short_memory(
                    state_old, final_move, reward, state_new, jsonDataParse["done"]
                )

                # remember
                agent.remember(state_old, final_move, reward,
                            state_new, jsonDataParse["done"])
                ws.send(json.dumps(final_move))

            if jsonDataParse["done"] == True:
                print("Training HAS ENDED")
                # train long memory, plot result
                agent.train_long_memory()

                record = score
                agent.model.save()

                total_score += score
    except simple_websocket.ConnectionClosed:
        pass
    return ''


if __name__ == "__main__":
    app.run(host="localhost", port=2943)