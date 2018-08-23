var player_move = [];
var player_value = [];
var enemy_move = [];
var enemy_value = [];

function update_plot()
{
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

    var data = [trace1, trace2];
    Plotly.newPlot('myDiv', data);
}

function reqMonitoring()
{
    var xhr = new XMLHttpRequest();
    
        xhr.onload = function() 
        {
            if(xhr.status == 200) 
            {
                ret = JSON.parse(xhr.responseText);
                
                player_move = ret.player_agent_moves;
                player_value = ret.player_agent_values;
                enemy_move = ret.enemy_agent_moves;
                enemy_value = ret.enemy_agent_values;
                
                update_plot();
            }
        };
    
        xhr.open('GET', 'http://127.0.0.1:5000/monitoring', true);
        xhr.send();
}

setInterval(reqMonitoring, 500);//1000 is miliseconds
