function showValue(reload=false, show_skip=true) {
  var new_value = ""
  if (!reload) {
    new_value = $('input[name="value"]').val();
  }
  $.getJSON($SCRIPT_ROOT + '/add_value', {
    value: new_value
  }, function(data) {
    $("#value-list").empty();
    for (x in data.result) {
      if (0 !== data.result[x].name.length) {
        var value_id = data.result[x].id;

        var skip_motivation_button = "";
        if (show_skip) {
          skip_motivation_button = `<div class="col-sm-5">
              <button class="btn similar-motivations-button" id="${value_id}" disabled="true">
              Potential Related Message <span class="glyphicon glyphicon-forward"></span>
              </button>
            </div>`
        }
        $("#value-list").append(
          ` <li class="list-group-item">
              <div class="row">
                <div class="col-sm-7">
                  <h3 style='display:inline-block'>
                    ${data.result[x].name}
                    <span class="glyphicon glyphicon-remove remove_value remove-entry-icon" id="${value_id}"></span>
                  </h3>
                </div>
              </div>
              <ul class="list-inline list-group-horizontal-sm" id='keyword-list${value_id}'>
              </ul>
              <br/>
              <div class="row">
                <div class="col-sm-12 form-group">
                  <div class="col-sm-8">
                    <form class="form-inline" action="javascript:void(0);">
                    <input class="form-control" placeholder="Enter a new keyword" type=text name="keyword_add${value_id}">
                    <input class="btn btn-default add_keyword" type="button" style="cursor: pointer" id="${value_id}" value="Add keyword">
                    </form>
                  </div>
                  <div class="col-sm-4">
                    <button type="button" class="btn btn-secondary open-dg-popup-button" id="${value_id}" data-toggle="modal" data-target="#addDefiningGoalPopup">
                       Add Description
                    </button>
                  </div>
                  <div class="col-sm-4">
                    <p class="form-control-static" id="keyword-suggestions${value_id}">
                    </p>
                  </div>
                </div>
              </div>
            </li>`
        );
        getKeywords(event=undefined, valud_id=value_id, trigger=false);
        if (value_id === data.suggestions.id) {
          $('#keyword-suggestions'+value_id).text("Similar terms: " + data.suggestions.text);
        }
      }
    }
    $('input[name="value"]').val('');
    if (!data.changed) {
      $(document).trigger("annotation_update", [{prevent_button_update: true}]);
    } else {
      $(document).trigger("annotation_update", [{prevent_button_update: false}]);
    }
  });
  return false;
}

function showKeywords(value_id, keywords, prefix="") {
    for (const i in keywords) {
        $("#" + prefix + "keyword-list" + value_id).append(
            ` <li class="text-center list-group-item"> ${keywords[i].name}
                <span class="glyphicon glyphicon-remove remove_keyword remove-entry-icon" id="${keywords[i].id}" ></span>
              </li>`
          );
    }
}

function getValuesCSD() {
    $("#value-list").empty();
    $.getJSON($SCRIPT_ROOT + '/csd_get_values', {}, function(data) {
        for (const x in data.values) {
            showValue(data.values[x]);
        }
    }).then(function (data) {
        for (const x in data.values) {
            getKeywordsCSD(data.values[x].id).then(function(kw_data) {
                showKeywords(data.values[x].id, kw_data.keywords);
            });
        }
    });
}

function getKeywordsCSD(value_id) {
    return $.getJSON($SCRIPT_ROOT + '/csd_get_keywords', {value_id: value_id});
}

function addValueCSD(event) {
    var new_value = $('input[id="add-value-input"]').val();
    $.getJSON($SCRIPT_ROOT + '/csd_add_value', {
        value: new_value
      }, function(data) {
        if (data.success) {
            showValue(data.value_obj);
        }
      }).then(function () {
          $('input[id="add-value-input"]').val('');
      });
}

function removeValueCSD(event) {
    var value_id = event.target.id;
    $.getJSON($SCRIPT_ROOT + '/csd_remove_value', {
        value_id: value_id,
      }, function(data) {
        if (data.success === true) {
            event.target.closest('li').remove();
            if ($('.left-value').prop('id') == value_id || $('.right-value').prop('id') == value_id) {
                $("#merge-value-name-field").prop('disabled', true);
                $('#submit').prop('disabled', true);
                $('#left').css('background-color', 'rgba(0,0,0,0.3)');
                $('#right').css('background-color', 'rgba(0,0,0,0.3)');
                $("#left .remove-entry-icon").remove();
                $("#right .remove-entry-icon").remove();
            }
        }
      });
}

function addKeywordCSD(event, check_other=true) {
    var value_id = event.currentTarget.id;
    var keyword_name = $('input[name=\"keyword_add'+value_id+'\"]').val();
    $.getJSON($SCRIPT_ROOT + '/csd_add_keyword', {
      keyword_name: keyword_name,
      value_id: value_id
    }, function(data) {
        if (data.success) {
            showKeywords(value_id, [data.keyword_obj]);
            if (check_other) {
                showKeywords(value_id, [data.keyword_obj], "merge-");
            }
        }
    }).then(function() {
        $('input[name="keyword_add' + value_id + '"]').val('');
    });
  }

function removeKeywordCSD(event, check_other=true) {
    var keyword_id = event.target.id;

    $.getJSON($SCRIPT_ROOT + '/csd_remove_keyword', {
        keyword_id: keyword_id,
    }, function(data) {
        if (data.success === true) {
            event.target.closest('li').remove();
            if (check_other) {
                // Check other occurrences of the keyword
                var x = $(".remove_keyword").closest('#'+keyword_id).closest('li').remove();
            }
        }
    });
}

function mergeValuesCSD(event) {
    var new_value = $('input[name="value"]').val();
    var left_value_id = $(".left-value").attr('id');
    var right_value_id = $(".right-value").attr('id');
    $.getJSON($SCRIPT_ROOT + '/csd_merge_pair', {
        value_id_0: left_value_id,
        value_id_1: right_value_id,
        merged_value_name: new_value
    }, function(data) {
        if (data.success) {
            $(document).trigger("consolidation_action", [{prevent_button_update: false}]);
            $("#merge-value-name-field").closest('.form-group').find('ul').prop('id','postmerge-keyword-list' + data.merged_value.id);
            showKeywords(data.merged_value.id, data.merged_keywords, 'postmerge-')
        }
    });
}

function switchButtons() {
    $('#submit').prop('disabled', true);
    $('#no-merge').prop('disabled', true);
    $('#next-pair').prop('disabled', false);
    $('#pick-pair').prop('disabled', false);
    $('#left').css('background-color', 'rgba(0,0,0,0.3)');
    $('#right').css('background-color', 'rgba(0,0,0,0.3)');
    $("#left .remove-entry-icon").remove();
    $("#right .remove-entry-icon").remove();
    $("#merge-value-name-field").prop('disabled', true);
}

function skipPairCSD(event) {
    $.getJSON($SCRIPT_ROOT + '/csd_skip_pair', {}, function(data) {
        $(document).trigger("consolidation_action");
    });
}

function updateValuePicker() {
    $('#value-picker-left').empty();
    $('#value-picker-right').empty();
    $.getJSON($SCRIPT_ROOT + '/csd_get_values', {}, function(data) {
        for (const x in data.values) {
            $('#value-picker-left').append(`<li><a href="#" id="${data.values[x].id}">${data.values[x].name}</a></li>`)
            $('#value-picker-right').append(`<li><a href="#" id="${data.values[x].id}">${data.values[x].name}</a></li>`)
        }
    })
}

function preloadNextPair() {
    var value_id_0 = $('.value-selection-btn-left').prop('id')
    var value_id_1 = $('.value-selection-btn-right').prop('id')
    $.getJSON($SCRIPT_ROOT + '/csd_preload_next_couple', {
        value_id_0: value_id_0,
        value_id_1: value_id_1
    }, function(data) {
        if (data.success) {
            window.location.reload();
        } else {
            if ($('#pickNextPairModalHeader').find('.alert').length == 0) {
                $('#pickNextPairModalHeader').append(`<div class="alert alert-danger">Unable to load this pair of values, perhaps you have already seen it?</div>`)
            }
        }
    });
}

function getDefiningGoal(event) {
    var value_id = event.target.id;
    $.getJSON($SCRIPT_ROOT + '/get_defining_goal', {
        value_id: value_id
    }, function(data) {
        if (data.success) {
            $('#defining-goal-textarea').val(data.defining_goal);
            $('.defining-goal-submit').prop('id', value_id);
            var value_name = $(event.target).closest('li').find('h3').text();
            $('#addDefiningGoalPopupTitle').text("Add description for: " + value_name);
        }
    });
}

function addDefiningGoal(event) {
    var value_id = event.target.id;
    var dg = $('#defining-goal-textarea').val();
    $.getJSON($SCRIPT_ROOT + '/add_defining_goal', {
        value_id: value_id,
        defining_goal: dg
    }, function(data) {
        if (data.success) {
            if ($('#addDefiningGoalPopupHeader').find('.alert').length > 0) {
                $('#addDefiningGoalPopupHeader').find('.alert').remove();
            }
        } else {
            if ($('#addDefiningGoalPopupHeader').find('.alert').length == 0) {
                $('#addDefiningGoalPopupHeader').append(`<div class="alert alert-danger">Unable to add defining goal!</div>`);
            }
        }
    });
}

function drawHistoryCSD() {
    var margin = {top: 50, right: 0, bottom: 200, left: 70};
    var width = 640;
    var height = width;
    // Keys of the legend, same order as csd_color_map in app/utils/backbone/utils.py
    var keys = [
        "Value addition",
        "Keyword addition",
        "Multiple additions",
        "Value removal",
        "Keyword removal",
        "Multiple removals",
        "Multiple additions and removals",
        "Values merge",
        "No actions taken"
    ];
    // Colors of the legend keys, as per csd_color_map in app/utils/backbone/utils.py
    var legend_colors = [
        '#31a354',  // Dark green
        '#a1d99b',  // Light green
        '#74c476',  // Green
        '#e6550d',  // Dark red
        '#fdae6b',  // Light red
        '#fd8d3c',  // Red
        '#9e9ac8',  // Purple
        '#756bb1',  // Dark Purple
        '#6baed6'   // Blue
    ];

    $.getJSON($SCRIPT_ROOT + '/csd_get_history', {
    }, function(data) {
        // Check if there is data in the history other than null
        var otherThanNull = data['history'].some(function (el) {
            return el !== null;
        });
        if (!otherThanNull) {
            return;
        }

        d3.select('#history-plot').remove();
        // Parse data to more suitable format
        var x_data = data['history'][0];
        var y_data = data['history'][1];
        var colors = data['history'][2];
        var zipped_data = x_data.map(function(_, i) {
            return [x_data[i], y_data[i], colors[i]];
        });

        // Set scales
        var x = d3.scaleBand()
            .range([0, width])
            .domain(d3.range(0, x_data.length))
            .rangeRound([0, width])
            .paddingInner(0.1)
            .paddingOuter(0.1);
        var y = d3.scaleLinear()
            .range([height, 0])
            .domain([0, d3.max(y_data)]);

    });
}
