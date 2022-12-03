current_activity= {
            "cough": 0.0,
            "falling": 78.8,
            "pain": 0.02,
            "sitting": 7.03,
            "sleeping": 0.0,
            "standing": 4.71,
            "walking": 9.43
        }
print({k: v for k, v in sorted(current_activity.items(), key=lambda item: item[1],reverse=True)})
print(max(current_activity, key=current_activity.get))