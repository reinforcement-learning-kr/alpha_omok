var baseurl = "http://220.83.128.81:5000/"

var c = document.getElementById("board");
var ctx = c.getContext("2d");

var game_board_size = 9;
var radius = 14;
var blank = 22;
var turn = 1; // 1 black 2 white
var prev_turn = -1;
var width = (game_board_size - 1) * 32 + blank * 2;
var height = (game_board_size - 1) * 32 + blank * 2;

var selecting_board_row = -1;
var selecting_board_col = -1;

var selected_board_row = -1;
var selected_board_col = -1;

var select_player_agent_name = document.getElementById('select_player_agent_name').options[0];
var select_enemy_agent_name = document.getElementById('select_enemy_agent_name').options[0];

var player_message = document.getElementById("player_message");
var enemy_message = document.getElementById("enemy_message");

var boardArray = new Array(game_board_size);
for (var i = 0; i < game_board_size; i++) {
	boardArray[i] = new Array(game_board_size);
	for (j = 0; j < game_board_size; j++) {
		boardArray[i][j] = 0;
	}
}

/* Mouse Event */
function getBoardPos(canvas, evt) {
	var rect = canvas.getBoundingClientRect();

	var xPos = evt.clientX - rect.left;
	var yPos = evt.clientY - rect.top;


	var xIdx = (xPos - blank) / 32;
	var resultX = Math.round(xIdx);
	var yIdx = (yPos - blank) / 32;
	var resultY = Math.round(yIdx);

	return {
		x: resultX,
		y: resultY
	};
}

c.addEventListener('mousemove', function (evt) {

	var pos = getBoardPos(c, evt);

	selecting_board_row = pos.y;
	selecting_board_col = pos.x;

	renderBoard();

}, false);

c.addEventListener('mousedown', function (evt) {
	var pos = getBoardPos(c, evt);

	if (boardArray[pos.x][pos.y] == 0) {
		selected_board_row = pos.y;
		selected_board_col = pos.x;
	}

	renderBoard();

}, false);

var headmap_color_blues = [
	"#f7fbff",
	"#deebf7",
	"#c6dbef",
	"#9ecae1",
	"#6baed6",
	"#4292c6",
	"#2171b5",
	"#08519c",
	"#08306b"
];

var headmap_color_greens = [
	"#f7fcf5",
	"#e5f5e0",
	"#c7e9c0",
	"#a1d99b",
	"#74c476",
	"#41ab5d",
	"#238b45",
	"#006d2c",
	"#00441b"
];

var heatmap_color_grays = [
	"#ffffff",
	"#f0f0f0",
	"#d9d9d9",
	"#bdbdbd",
	"#969696",
	"#737373",
	"#525252",
	"#252525",
	"#000000"
];

var heatmap_color_oranges = [
	"#fff5eb",
	"#fee6ce",
	"#fdd0a2",
	"#fdae6b",
	"#fd8d3c",
	"#f16913",
	"#d94801",
	"#a63603",
	"#7f2704"
];

var heatmap_color_purples = [
	"#fcfbfd",
	"#efedf5",
	"#dadaeb",
	"#bcbddc",
	"#9e9ac8",
	"#807dba",
	"#6a51a3",
	"#54278f",
	"#3f007d"
];

var heatmap_color_reds = [
	"#fff5f0",
	"#fee0d2",
	"#fcbba1",
	"#fc9272",
	"#fb6a4a",
	"#ef3b2c",
	"#cb181d",
	"#a50f15",
	"#67000d"
];

var player_move = [];
var player_value = [];
var enemy_move = [];
var enemy_value = [];

var action_index = -1;

function updateBoard(ret) {
	if (ret.curr_turn == 0) // black turn: 0, white turn: 1
	{
		turn = 1 // black
	}
	else {
		turn = 2 // white
	}

	// if turn change
	if (prev_turn != turn) {
		selecting_board_row = -1;
		selecting_board_col = -1;
		selected_board_row = -1;
		selected_board_col = -1;

		prev_turn = turn;
	}

	action_index = ret.action_index;

	game_board_size = ret.game_board_size;

	for (var i = 0; i < game_board_size; i++) {
		for (j = 0; j < game_board_size; j++) {
			idx = i + j * game_board_size;

			switch (ret.game_board_values[idx]) {
				case 1:
					boardArray[i][j] = 1;
					break;
				case -1:
					boardArray[i][j] = 2;
					break;
				case 0:
					boardArray[i][j] = 0;
					break;
			}
		}
	}
}

function updateVPlot(ret) {
	player_move = ret.player_agent_moves;
	player_value = ret.player_agent_values;
	enemy_move = ret.enemy_agent_moves;
	enemy_value = ret.enemy_agent_values;
}

function renderBoard() {

	// board fill color
	ctx.fillStyle = "#ffcc66";
	ctx.fillRect(0, 0, width, height);

	// board draw line
	ctx.strokeStyle = 'black';
	ctx.fillStyle = "#FF0000";
	ctx.lineWidth = 1
	for (i = 0; i < game_board_size; i++) {
		// horizontal line draw
		ctx.beginPath();
		ctx.moveTo(blank + i * 32 + 0.5, blank);
		ctx.lineTo(blank + i * 32 + 0.5, width - blank + 0.5);
		ctx.stroke();

		// vertical line draw
		ctx.beginPath();
		ctx.moveTo(blank, blank + i * 32 + 0.5);
		ctx.lineTo(height - blank, blank + i * 32 + 0.5);
		ctx.stroke();
	}

	ctx.shadowColor = 'black';

	// console.log('render : ' + turn);

	// board draw
	for (i = 0; i < game_board_size; i++) {
		for (j = 0; j < game_board_size; j++) {
			ctx.shadowBlur = 5;
			ctx.shadowOffsetX = 2;
			ctx.shadowOffsetY = 2;
			ctx.shadowColor = "#333333";

			if (boardArray[i][j] == 1) {
				ctx.beginPath();
				ctx.strokeStyle = "#000000";
				ctx.fillStyle = "#000000";
				ctx.arc(blank + i * 32, blank + j * 32, radius, 0, 2 * Math.PI);
				ctx.fill();
				//ctx.stroke();
			} else if (boardArray[i][j] == 2) {
				ctx.beginPath();
				ctx.strokeStyle = "#ffffff";
				ctx.fillStyle = "#ffffff";
				ctx.arc(blank + i * 32, blank + j * 32, radius, 0, 2 * Math.PI);
				ctx.fill();
				//ctx.stroke();
			}

			ctx.shadowOffsetX = 0;
			ctx.shadowOffsetY = 0;
			ctx.shadowBlur = 0;

			ctx.lineWidth = 2;

			if (action_index == (j * game_board_size + i)) {
				if (boardArray[i][j] == 1) {
					ctx.beginPath();
					ctx.strokeStyle = "#ffffff";
					ctx.arc(blank + i * 32, blank + j * 32, radius / 2.0, 0, 2 * Math.PI);
					ctx.stroke();
				} else if (boardArray[i][j] == 2) {
					ctx.beginPath();
					ctx.strokeStyle = "#000000";
					ctx.arc(blank + i * 32, blank + j * 32, radius / 2.0, 0, 2 * Math.PI);
					ctx.stroke();
				}
			}

			if (i == selecting_board_row && j == selecting_board_col) {
				ctx.lineWidth = 1;

				ctx.beginPath();
				ctx.globalAlpha = 0.8;

				if (turn == 1) // 1 black
				{
					ctx.strokeStyle = "#000000";
					ctx.fillStyle = "#000000";
				}
				else if (turn == 2) // 2 white
				{
					ctx.strokeStyle = "#ffffff";
					ctx.fillStyle = "#ffffff";
				}

				ctx.arc(blank + selecting_board_col * 32, blank + selecting_board_row * 32, radius, 0, 2 * Math.PI);
				ctx.fill();
				ctx.stroke();
				ctx.globalAlpha = 1;
			}

			if (i == selected_board_row && j == selected_board_col) {
				ctx.lineWidth = 5;

				ctx.beginPath();

				if (turn == 1) // 1 black
				{
					ctx.strokeStyle = "#000000";
					ctx.fillStyle = "#aaaaaa";
				}
				else if (turn == 2) // 2 white
				{
					ctx.strokeStyle = "#ffffff";
					ctx.fillStyle = "#aaaaaa";
				}

				ctx.arc(blank + selected_board_col * 32, blank + selected_board_row * 32, radius - 2, 0, 2 * Math.PI);
				ctx.stroke();
			}
		}
	}

	// board_message.innerHTML = "AlphaOmoc vs AlphaOmoc";
}

function renderVPlot() {
	var trace1 = {
		x: player_move,
		y: player_value,
		type: 'scatter',
		fill: 'tozeroy',
		name: 'player'
	};

	var trace2 = {
		x: enemy_move,
		y: enemy_value,
		type: 'scatter',
		fill: 'tozeroy',
		name: 'enemy'
	};

	var layout = {
		title: 'Value Network',
		xaxis: {
			range: [0, 81],
			autorange: false
		},
		yaxis: {
			range: [0.0, 100],
			autorange: false
		}
	};

	var data = [trace1, trace2];
	Plotly.newPlot('v_monitoring', data, layout);
}

function reqPeriodicStatus() {
	var xhr = new XMLHttpRequest();

	xhr.onload = function () {
		if (xhr.status == 200) {
			ret = JSON.parse(xhr.responseText);

			select_player_agent_name.value = ret.player_agent_name;
			select_player_agent_name.text = ret.player_agent_name;
			select_enemy_agent_name.value = ret.enemy_agent_name;
			select_enemy_agent_name.text = ret.enemy_agent_name;

			updateBoard(ret);
			updateVPlot(ret);

			renderBoard();
			renderVPlot();
		}
	};

	xhr.open('GET', baseurl + 'periodic_status', true);
	xhr.send();
}

function reqPromptStatus() {
	var xhr = new XMLHttpRequest();

	xhr.onload = function () {
		if (xhr.status == 200) {
			ret = JSON.parse(xhr.responseText);

			player_message.innerHTML = 'player : ' + ret.player_message;
			enemy_message.innerHTML = 'enemy : ' + ret.enemy_message;
		}
	};

	xhr.open('GET', baseurl + 'prompt_status', true);
	xhr.send();
}

function reqResetAgents() {
	var xhr = new XMLHttpRequest();

	xhr.onload = function () {
		if (xhr.status == 200) {
			document.getElementById("select_player_agent_name").options[0].selected = true;
			document.getElementById("select_enemy_agent_name").options[0].selected = true;
		}
	};

	selected_player_agent_name = document.getElementById("select_player_agent_name").value;
	selected_enemy_agent_name = document.getElementById("select_enemy_agent_name").value;

	xhr.open('GET', baseurl + 'req_reset_agenets?player_agent=' + selected_player_agent_name + '&enemy_agent=' + selected_enemy_agent_name, true);
	xhr.send();
}

function reqMove() {
	if (selected_board_col >= 0
		&& selected_board_col < game_board_size
		&& selected_board_row >= 0
		&& selected_board_row < game_board_size
		&& boardArray[selected_board_col][selected_board_row] == 0) {
		action_idx = selected_board_col + selected_board_row * game_board_size;

		reqAction(action_idx)
	}
}

function reqAction(action_idx) {
	var xhr = new XMLHttpRequest();

	xhr.onload = function () {
		if (xhr.status == 200) {

		}
	};

	xhr.open('GET', baseurl + 'action?action_idx=' + action_idx.toString(), true);
	xhr.send();
}

function clearBoard() {
	for (var i = 0; i < game_board_size; i++) {
		for (j = 0; j < game_board_size; j++) {
			boardArray[i][j] = 0;
		}
	}

}

renderBoard();
renderVPlot();

setInterval(reqPeriodicStatus, 500);
setInterval(reqPromptStatus, 100);
