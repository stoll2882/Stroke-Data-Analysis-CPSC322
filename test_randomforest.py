import numpy as np
import scipy.stats as stats 

from myclassifiers import MyRandomForestClassifier, MyDecisionTreeClassifier

def test_random_forest_classifier_fit():
    # DECISION TREE TEST

    #interview_header = ["level", "lang", "tweets", "phd", "interviewed_well"]
    #header = ["att0","att1","att2","att3","class"]
    interview_table = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]

    interview_results = ["False", "False","True","True","True","False","True","False","True","True","True","True","True","False"]
    #tree we constructed in class
    interview_tree = \
        ["Attribute", "att0",
            ["Value", "Junior", 
                ["Attribute", "att3",
                    ["Value", "no", 
                        ["Leaf", "True", 3, 5]
                    ],
                    ["Value", "yes", 
                        ["Leaf", "False", 2, 5]
                    ]
                ]
            ],
            ["Value", "Mid",
                ["Leaf", "True", 4, 14]
            ],
            ["Value", "Senior",
                ["Attribute", "att2",
                    ["Value", "no",
                        ["Leaf", "False", 3, 5]
                    ],
                    ["Value", "yes",
                        ["Leaf", "True", 2, 5]
                    ]
                ]
            ]
        ]

    classifier = MyDecisionTreeClassifier()
    classifier.fit(interview_table, interview_results)
    assert np.allclose(classifier.tree, interview_tree)
    # only decision tree atm