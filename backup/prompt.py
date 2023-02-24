"""
Pretraining Tasks -- 5 Prompt Families (1, 2, 3, 4, 5)
"""

# =====================================================
# Task Subgroup 1 -- Sequential item -- 10 Prompts
# =====================================================

task_subgroup_1 = []

template = {}
template[
    "source"
] = "User_{} has purchased {} , predict next possible item to be bought by the user ?"
template["target"] = "{}"
template["input_first"] = "user"
template["task"] = "sequential"
template["id"] = "1-1"

task_subgroup_1.append(template)


template = {}
template[
    "source"
] = "I find the purchase list of user_{} : {} , I wonder what other itmes does the user need . Can you help me decide ?"
template["target"] = "{}"
template["input_first"] = "user"
template["task"] = "sequential"
template["id"] = "1-2"

task_subgroup_1.append(template)


template = {}
template[
    "source"
] = "Here is the purchase list of user_{} : {} , try to recommend another item to the user ."
template["target"] = "{}"
template["input_first"] = "user"
template["task"] = "sequential"
template["id"] = "1-3"

task_subgroup_1.append(template)


template = {}
template[
    "source"
] = "According to what items user_{} has purchased : {} , Can you recommend another item to the user ?"
template["target"] = "{}"
template["input_first"] = "user"
template["task"] = "sequential"
template["id"] = "1-4"

task_subgroup_1.append(template)


template = {}
template[
    "source"
] = "The user_{} has bought items : {} , What else do you think is necessary for the user ?"
template["target"] = "{}"
template["input_first"] = "user"
template["task"] = "sequential"
template["id"] = "1-5"

task_subgroup_1.append(template)


template = {}
template[
    "source"
] = "Here is the item purchase history of user_{} : {} , What to recommend next for the user ?"
template["target"] = "{}"
template["input_first"] = "user"
template["task"] = "sequential"
template["id"] = "1-6"

task_subgroup_1.append(template)


template = {}
template["source"] = "What would user_{} be likely to purchase next after buying {} ?"
template["target"] = "{}"
template["input_first"] = "user"
template["task"] = "sequential"
template["id"] = "1-7"

task_subgroup_1.append(template)


template = {}
template[
    "source"
] = "By analyzing the user_{} 's purchase of {}, what is the next item expected to be bought ?"
template["target"] = "{}"
template["input_first"] = "user"
template["task"] = "sequential"
template["id"] = "1-8"

task_subgroup_1.append(template)


template = {}
template[
    "source"
] = "After buying items {} , what is the next item that could be recommended for user_{} ?"
template["target"] = "{}"
template["input_first"] = ""
template["task"] = "sequential"
template["id"] = "1-9"

task_subgroup_1.append(template)


template = {}
template[
    "source"
] = "Can you recommend the next item for user_{} , given the user 's purchase of {} ?"
template["target"] = "{}"
template["input_first"] = "user"
template["task"] = "sequential"
template["id"] = "1-10"

task_subgroup_1.append(template)


# =====================================================
# Task Subgroup 2 -- Sequential Yes/no -- 8 Prompts
# =====================================================
task_subgroup_2 = []
# Pairwise Prediction
template = {}
template[
    "source"
] = "User_{} has bought the following items : {} , does the user likely to buy {} as well ?"
template["target"] = "{}"
template["input_first"] = "user"
template["task"] = "sequential_yesno"
template["id"] = "2-1"

task_subgroup_2.append(template)


template = {}
template[
    "source"
] = "According to user_{} 's bought item list : {} , Predict whether the user will buy or has already bought {} ."
template["target"] = "{}"
template["input_first"] = "user"
template["task"] = "sequential_yesno"
template["id"] = "2-2"

task_subgroup_2.append(template)


template = {}
template[
    "source"
] = "According to user_{} 's bought item list : {} , Do you think the user also needs {} ?"
template["target"] = "{}"
template["input_first"] = "user"
template["task"] = "sequential_yesno"
template["id"] = "2-3"

task_subgroup_2.append(template)


# =====================================================
# Task Subgroup 3 -- Direct -- 5 Prompts
# =====================================================

task_subgroup_3 = []

template = {}
template["source"] = "Will user_{} likely to purchase item_{} ?"
template["target"] = "{}"
template["input_first"] = "user"
template["task"] = "direct_yesno"
template["target_argv"] = ["yes_no"]
template["id"] = "3-1"

task_subgroup_3.append(template)


template = {}
template["source"] = "Shall we recommend item_{} to user_{} ?"
template["target"] = "{}"
template["input_first"] = "item"
template["task"] = "direct_yesno"
template["target_argv"] = ["yes_no"]
template["id"] = "3-2"

task_subgroup_3.append(template)


template = {}
template["source"] = "For user_{}, do you think it is good to recommend item_{} ?"
template["target"] = "{}"
template["input_first"] = "user"
template["task"] = "direct_yesno"
template["target_argv"] = ["yes_no"]
template["id"] = "3-3"

task_subgroup_3.append(template)


template = {}
template[
    "source"
] = "I would like to recommend some items for user_{} . Is the following item_{} a good choice ?"
template["target"] = "{}"
template["input_first"] = "user"
template["task"] = "direct_yesno"
template["target_argv"] = ["yes_no"]
template["id"] = "3-4"

task_subgroup_3.append(template)


template = {}
template["source"] = "Do you think user_{} will like to buy item_{} ?"
template["target"] = "{}"
template["input_first"] = "user"
template["task"] = "direct_yesno"
template["target_argv"] = ["yes_no"]
template["id"] = "3-5"

task_subgroup_3.append(template)


# =====================================================
# Task Subgroup 4 -- Direct Candidates -- 5 Prompts
# =====================================================
task_subgroup_4 = []

template = {}
template["source"] = "Which one of the following items to recommend for user_{} ? {}"
template["target"] = "{}"
template["input_first"] = "user"
template["task"] = "direct_candidates"
template["target_argv"] = ["groundtruth_item_ids"]
template["id"] = "4-1"

task_subgroup_4.append(template)


template = {}
template[
    "source"
] = "Choose the best item from the candidates to recommend for user_{} ? {}"
template["target"] = "{}"
template["input_first"] = "user"
template["task"] = "direct_candidates"
template["target_argv"] = ["groundtruth_item_ids"]
template["id"] = "4-2"

task_subgroup_4.append(template)


template = {}
template[
    "source"
] = "Pick the most useful item from the following list and recommend to user_{} : {}"
template["target"] = "{}"
template["input_first"] = "user"
template["task"] = "direct_candidates"
template["target_argv"] = ["groundtruth_item_ids"]
template["id"] = "4-3"

task_subgroup_4.append(template)


template = {}
template[
    "source"
] = "We want to make recommendation for user_{} .  Select the best item from these candidates : {}"
template["target"] = "{}"
template["input_first"] = "user"
template["task"] = "direct_candidates"
template["target_argv"] = ["groundtruth_item_ids"]
template["id"] = "4-4"

task_subgroup_4.append(template)


template = {}
template[
    "source"
] = "Our intention is to select the best item for user_{} from the below candidates : {}"
template["target"] = "{}"
template["input_first"] = "user"
template["task"] = "direct_candidates"
template["target_argv"] = ["groundtruth_item_ids"]
template["id"] = "4-5"

task_subgroup_4.append(template)


# ========================================================
# Task Subgroup 5 -- Direct Straightforward -- 5 Prompts
# ========================================================
task_subgroup_5 = []


template = {}
template["source"] = "What should we recommend for user_{} ?"
template["target"] = "{}"
template["input_first"] = "user"
template["task"] = "direct_straightforward"
template["id"] = "5-1"
task_subgroup_5.append(template)


template = {}
template["source"] = "Which recommendation should we provide to user_{} ?"
template["target"] = "{}"
template["input_first"] = "user"
template["task"] = "direct_straightforward"
template["id"] = "5-2"

task_subgroup_5.append(template)

template = {}
template["source"] = "How can we assist user_{} with a recommendation ?"
template["target"] = "{}"
template["input_first"] = "user"
template["task"] = "direct_straightforward"
template["id"] = "5-3"

task_subgroup_5.append(template)

template = {}
template["source"] = "What would be a suitable recommendation for user_{} ?"
template["target"] = "{}"
template["input_first"] = "user"
template["task"] = "direct_straightforward"
template["id"] = "5-4"

task_subgroup_5.append(template)

template = {}
template["source"] = "What would be a helpful recommendation for user_{} ?"
template["target"] = "{}"
template["input_first"] = "user"
template["task"] = "direct_straightforward"
template["id"] = "5-5"

task_subgroup_5.append(template)


# ========================================================
# Task Subgroup 6 -- Meta Data Title -- 5 Prompts
# ========================================================
task_subgroup_6 = []


template = {}
template["source"] = "Which item has the title {} ?"
template["target"] = "{}"
template["task"] = "title"
template["id"] = "6-1"
task_subgroup_6.append(template)


template = {}
template["source"] = "Which item is given the name of {} ?"
template["target"] = "{}"
template["task"] = "title"
template["id"] = "6-2"
task_subgroup_6.append(template)


template = {}
template["source"] = "An item is called {} . Which item is it ?"
template["target"] = "{}"
template["task"] = "title"
template["id"] = "6-3"
task_subgroup_6.append(template)


template = {}
template["source"] = "Which item is called {} ?"
template["target"] = "{}"
template["task"] = "title"
template["id"] = "6-4"
task_subgroup_6.append(template)


template = {}
template["source"] = "One item goes by the name {} . Can you tell me which item it is ?"
template["target"] = "{}"
template["task"] = "title"
template["id"] = "6-5"
task_subgroup_6.append(template)


# ================================================================================
# Task Subgroup 9 -- Meta Data use description to find item -- 5 Prompts
# ================================================================================
task_subgroup_9 = []

template = {}
template[
    "source"
] = "An item can be described as follows: {} . Which item is it describing ?"
template["target"] = "{}"
template["task"] = "description"
template["input"] = "description"
template["id"] = "9-1"
task_subgroup_9.append(template)


template = {}
template["source"] = "Can you tell me which item is described as {} ?"
template["target"] = "{}"
template["task"] = "description"
template["input"] = "description"
template["id"] = "9-2"
task_subgroup_9.append(template)


template = {}
template[
    "source"
] = "Can you provide the name of the item that is considered as this ? {} "
template["target"] = "{}"
template["task"] = "description"
template["input"] = "description"
template["id"] = "9-3"
task_subgroup_9.append(template)


template = {}
template["source"] = "Which item has been described as below ? Description : {} "
template["target"] = "{}"
template["task"] = "description"
template["input"] = "description"
template["id"] = "9-4"
task_subgroup_9.append(template)


template = {}
template["source"] = "What item fits the below description ? Description : {}"
template["target"] = "{}"
template["task"] = "description"
template["input"] = "description"
template["id"] = "9-5"
task_subgroup_9.append(template)


template = {}
template["source"] = "Which item has the following characteristics ? {}"
template["target"] = "{}"
template["task"] = "description"
template["input"] = "description"
template["id"] = "9-6"
task_subgroup_9.append(template)


template = {}
template[
    "source"
] = "Which item possesses the quality of described as follows ? Description : {}"
template["target"] = "{}"
template["task"] = "description"
template["input"] = "description"
template["id"] = "9-7"
task_subgroup_9.append(template)


template = {}
template[
    "source"
] = "I am curious to know which item can be described as below: {} .  Can you tell me ?"
template["target"] = "{}"
template["task"] = "description"
template["input"] = "description"
template["id"] = "9-8"
task_subgroup_9.append(template)


# ========================================================
# Task Subgroup 8 -- Meta Data Category -- 5 Prompts
# ========================================================
task_subgroup_8 = []


template = {}
template["source"] = "What categories does item_{} belong to ?"
template["target"] = "{}"
template["task"] = "category"
template["id"] = "8-1"
task_subgroup_8.append(template)


template = {}
template["source"] = "To which categories does item_{} pertain ?"
template["target"] = "{}"
template["task"] = "category"
template["id"] = "8-2"
task_subgroup_8.append(template)


template = {}
template["source"] = "Which category or categories does item_{} fit into ?"
template["target"] = "{}"
template["task"] = "category"
template["id"] = "8-3"
task_subgroup_8.append(template)


template = {}
template["source"] = "In what categories is item_{} classified ?"
template["target"] = "{}"
template["task"] = "category"
template["id"] = "8-4"
task_subgroup_8.append(template)


template = {}
template["source"] = "What are item_{} 's category classifications ?"
template["target"] = "{}"
template["task"] = "category"
template["id"] = "8-5"
task_subgroup_8.append(template)


template = {}
template["source"] = "What is the nature of item_{} ?"
template["target"] = "{}"
template["task"] = "category"
template["id"] = "8-6"
task_subgroup_8.append(template)


template = {}
template["source"] = "How do you classify this product item_{} ?"
template["target"] = "{}"
template["task"] = "category"
template["id"] = "8-7"
task_subgroup_8.append(template)


# ================================================================
# Task Subgroup 10 -- Meta Data Reviews, ask item -- 6 Prompts
# ================================================================
task_subgroup_10 = []


template = {}
template[
    "source"
] = "User_{} made a comment for one item as this: {}  Can you tell me which item it is ?"
template["target"] = "{}"
template["input_first"] = "user"
template["task"] = "review"
template["id"] = "10-1"
task_subgroup_10.append(template)


template = {}
template[
    "source"
] = "Which item recived the view stated below from user_{} ? Review : {}"
template["target"] = "{}"
template["input_first"] = "user"
template["task"] = "review"
template["id"] = "10-2"
task_subgroup_10.append(template)


template = {}
template[
    "source"
] = "I am trying to find out which item was described as this : {} , which is presented by user_{} ."
template["target"] = "{}"
template["input_first"] = "review"
template["task"] = "review"
template["id"] = "10-3"
task_subgroup_10.append(template)


template = {}
template[
    "source"
] = "Can you identify the item that user_{} has described as below ? {}"
template["target"] = "{}"
template["input_first"] = "user"
template["task"] = "review"
template["id"] = "10-4"
task_subgroup_10.append(template)


template = {}
template[
    "source"
] = " Can you determine which item user_{} was referring to when they left a comment as below ? Review : {}"
template["target"] = "{}"
template["input_first"] = "user"
template["task"] = "review"
template["id"] = "10-5"
task_subgroup_10.append(template)


template = {}
template[
    "source"
] = " User_{} has left a comment below : {} I need to know which item it is describing . Can you help me find it  ? "
template["target"] = "{}"
template["input_first"] = "user"
template["task"] = "review"
template["id"] = "10-6"
task_subgroup_10.append(template)


# ================================================================
# Task Subgroup 11 -- Meta Data Reviews, ask user -- 6 Prompts
# ================================================================
task_subgroup_11 = []


template = {}
template[
    "source"
] = "Item_{} has received a comment as below : {} Can you tell me which user made this comment ?"
template["target"] = "{}"
template["input_first"] = "item"
template["task"] = "review"
template["id"] = "11-1"
task_subgroup_11.append(template)


template = {}
template["source"] = "Who left the comment below for item_{} ? Review : {}"
template["target"] = "{}"
template["input_first"] = "item"
template["task"] = "review"
template["id"] = "11-2"
task_subgroup_11.append(template)


template = {}
template[
    "source"
] = "Can you tell me who left the below comment for item_{} ? Comment: {}"
template["target"] = "{}"
template["input_first"] = "item"
template["task"] = "review"
template["id"] = "11-3"
task_subgroup_11.append(template)


template = {}
template["source"] = "Which user had such opinion on item_{} ? Opinion : {}"
template["target"] = "{}"
template["input_first"] = "item"
template["task"] = "review"
template["id"] = "11-4"
task_subgroup_11.append(template)


template = {}
template["source"] = "Who described item_{} as such ? Description : {}"
template["target"] = "{}"
template["input_first"] = "item"
template["task"] = "review"
template["id"] = "11-5"
task_subgroup_11.append(template)


template = {}
template[
    "source"
] = "I need to know which user made the below comment for item_{} : {} . Can you help me find the user ?"
template["target"] = "{}"
template["input_first"] = "item"
template["task"] = "review"
template["id"] = "11-6"
task_subgroup_11.append(template)
