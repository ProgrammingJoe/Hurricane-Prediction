from csv import DictReader
import re

class HurricancePrediction:

    def read(self, data_location):
        with open(data_location, 'r') as f:
            r = DictReader(f)
            hurricanes_dict = {}
            current_id = 0
            new_hurricane = []
            for row in r:
                if current_id == row['ID']:
                    new_hurricane.append({
                        'Date': row['Date'],
                        'Time': row['Time'],
                        'Lat': row['Latitude'],
                        'Long': row['Longitude'],
                        'MaxWind': row['Maximum Wind']
                    })
                else:
                    hurricanes_dict[current_id] = new_hurricane
                    current_id = row['ID']
                    new_hurricane = [{
                        'Date': row['Date'],
                        'Time': row['Time'],
                        'Lat': row['Latitude'],
                        'Long': row['Longitude'],
                        'MaxWind': row['Maximum Wind']
                    }]

            return hurricanes_dict

    def merge_rows(self, hurricanes):
        data = []
        for key in hurricanes:
            hurricane = hurricanes[key]
            for row in range(0, len(hurricane) - 1):
                if hurricane[row+1]:
                    data.append([int(hurricane[row]['Date']),
                        int(hurricane[row]['Time']),
                        self._lattoint(hurricane[row]['Lat']),
                        self._longtoint(hurricane[row]['Long']),
                        int(hurricane[row]['MaxWind']),
                        int(hurricane[row + 1]['Date']),
                        int(hurricane[row + 1]['Time']),
                        self._lattoint(hurricane[row + 1]['Lat']),
                        self._longtoint(hurricane[row + 1]['Long']),
                        int(hurricane[row + 1]['MaxWind'])
                    ])
        return data

    def _lattoint(self, latitude):
        if 'N' in latitude:
            return float(latitude[:-1]) + 90
        elif 'S' in latitude:
            return 90 - float(latitude[:-1])

    def _longtoint(self, longitude):
        if 'W' in longitude:
            return float(longitude[:-1]) + 180
        elif 'E' in longitude:
            return 180 - float(longitude[:-1])

def main():
    predictor = HurricancePrediction()

    pacific_hurricanes = predictor.read('data/pacific.csv')
    atlantic_hurricanes = predictor.read('data/atlantic.csv')

    pacific_training = predictor.merge_rows(pacific_hurricanes)
    print(pacific_training[0])
    print(pacific_training[1])


if __name__ == "__main__":
    main()
