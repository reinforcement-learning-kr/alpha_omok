import flask
from flask import Blueprint
from info.agent_info import AgentInfo
from info.game_info import GameInfo
import utils

# env_small: 9x9, env_regular: 15x15
from env import env_small as game
BOARD_SIZE = game.Return_BoardParams()[0]

web_api = Blueprint('web_api', __name__)

game_info = GameInfo(BOARD_SIZE)
player_agent_info = AgentInfo(BOARD_SIZE)
enemy_agent_info = AgentInfo(BOARD_SIZE)


@web_api.route('/gameboard')
def gameboard():
    return flask.render_template('gameboard.html')


@web_api.route('/dashboard')
def dashboard():
    return flask.render_template('dashboard.html')


@web_api.route('/periodic_status')
def periodic_status():

    data = {"success": False}

    data["game_board_size"] = game_info.game_board.shape[0]
    data["game_board_values"] = game_info.game_board.reshape(
        game_info.game_board.size).astype(int).tolist()
    data["game_board_message"] = game_info.message
    data["action_index"] = game_info.action_index
    data["win_index"] = game_info.win_index
    data["curr_turn"] = game_info.curr_turn
    data["enemy_turn"] = game_info.enemy_turn
    data["player_agent_name"] = player_agent_info.agent.get_name()
    data["enemy_agent_name"] = enemy_agent_info.agent.get_name()

    data["player_agent_p_size"] = player_agent_info.p_size
    data["player_agent_p_values"] = player_agent_info.p.reshape(
        player_agent_info.p_size).astype(float).tolist()
    data["player_agent_visit_size"] = player_agent_info.visit_size
    data["player_agent_visit_values"] = player_agent_info.visit.reshape(
        player_agent_info.visit_size).astype(float).tolist()

    data["enemy_agent_p_size"] = enemy_agent_info.p_size
    data["enemy_agent_p_values"] = enemy_agent_info.p.reshape(
        enemy_agent_info.p_size).astype(float).tolist()
    data["enemy_agent_visit_size"] = enemy_agent_info.visit_size
    data["enemy_agent_visit_values"] = enemy_agent_info.visit.reshape(
        enemy_agent_info.visit_size).astype(float).tolist()

    data["player_agent_moves"] = player_agent_info.moves
    data["player_agent_values"] = player_agent_info.values
    data["enemy_agent_moves"] = enemy_agent_info.moves
    data["enemy_agent_values"] = enemy_agent_info.values

    data["success"] = True

    return flask.jsonify(data)


@web_api.route('/prompt_status')
def prompt_status():
    data = {"success": False}

    data["player_message"] = player_agent_info.agent.get_message()
    data["enemy_message"] = enemy_agent_info.agent.get_message()
    data["success"] = True

    return flask.jsonify(data)
