var api_root_url = location.hostname=='localhost' ?  'http://localhost:5000' : 'https://dopelemon.me/api'
var res_folder_root_url = location.hostname=='localhost' ? '' : 'https://dopelemon.me/'


var pose_list = []
var filtered_results = []

var updateFilteredImgCount = function(len){
    $("#filtered-results-div-title").empty()
    $("#filtered-results-div-title").append('<label>'+len+' filtered poses</label>')
}

var apply_pose_type_filter = function(filtered_results, changedElement, pose_list, filterObj){
    if(changedElement.checked){
        filtered_results = filtered_results.filter(function(item){
            return item['pose_meta'].indexOf(changedElement.value)!=-1
        })
    }
    else{
        var res = Object.assign([], pose_list);
        if(filterObj['difficulty']!="Any"){
            var i = (filterObj['difficulty']=="Easy"?1: (filterObj['difficulty']=="Medium")?2:3)
            res = res.filter(function(item){
                return item['difficulty'] == i
            })
        }
        if(filterObj['pose_meta'].length){
            res = res.filter(function(item){
                return item['pose_meta'].filter(function(meta){
                    return filterObj['pose_meta'].indexOf(meta) > -1
                }).length > 0
            })
        }

        filtered_results = res
    }
    updateFilteredImgCount(filtered_results.length)
    return filtered_results
}

var filterObj = {
    'difficulty' : 'Easy',
    'pose_meta' : ['Forward Bend']
}

var appendFilteredImagesToDiv = function(filtered_results){
    $("#filtered-results-img-container").empty()

    filtered_results.forEach(function(item){
        $("#filtered-results-img-container").append('<div><div><img src="images/yoga-poses/asana_gt/'+item['sanskrit_name'].toLocaleLowerCase()
        +'.png"/></div><label>'+item['sanskrit_name']+' ('+item['english_name']+')</label></div>')
    })
}

$.getJSON('yoga-poses-data/pose-list-with-meta.json', function(json) {
    pose_list = json; 
    filtered_results = apply_pose_type_filter(filtered_results, {checked:false}, pose_list, filterObj)
    appendFilteredImagesToDiv(filtered_results)
});


var get_opt_group_data = function(group_label, options, def_opt){
    var res = {}
    res["type"] = "optiongroup"
    res["label"] = group_label
    res["children"] = []
    var counter = 0
    options.forEach(element => {
        res["children"].push({ "type": "option", "value": element, "label": element, "selected":element==def_opt})
        counter += 1
    });
    return res
}

$('#difficulty-select').searchableOptionList({
    data : [
        get_opt_group_data("", ["Any", "Easy", "Medium", "Hard"], "Easy")  
    ],
    maxHeight : '200px',
    showSelectAll : false,
    showSelectNone : false,
    events : {
        onChange : function(sol,changedElement){
            filterObj['difficulty'] = changedElement[0].value
            filtered_results = apply_pose_type_filter(filtered_results, {checked:false}, pose_list, filterObj)
            appendFilteredImagesToDiv(filtered_results)   
        }	
    }
});

$('#pose-type-select').searchableOptionList({
    data : [
        get_opt_group_data("", ["Hip Opening", "Seated", "Twist", "Forward Bend", "Standing", "Core", "Strengthening", "Chest Opening", "Backbends", "Restorative", "Arm Balance", "Balancing", "Inversion", "Binding", "Kneeling"], "Forward Bend")  
    ],
    maxHeight : '200px',
    showSelectAll : false,
    showSelectNone : false,
    events : {
        onChange : function(sol,changedElement){
            if(changedElement[0].checked){
                filterObj['pose_meta'].push(changedElement[0].value)
            }
            else{
                filterObj['pose_meta'].splice(filterObj['pose_meta'].indexOf(changedElement[0].value), 1)
            }
            
            filtered_results = apply_pose_type_filter(filtered_results, changedElement[0], pose_list, filterObj)
            appendFilteredImagesToDiv(filtered_results)   
        }	
    }
});


function invoke_upload_image(){
    $('#upload-btn').click();
} 

function append_resultant_image_to_dom(res_url){
    var res_dom_container_id = 'result'
    $('#'+res_dom_container_id).empty()
    $('#'+res_dom_container_id).append('<img src='+res_url+' />')
}


