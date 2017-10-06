from csv import DictReader

class HurricancePrediction:

    def read(self, data_location):
        with open(data_location, 'r') as f:
            r = DictReader(f)
            hurricanes_dict = {}
            current_id = 0
            new_hurricane = []
            hurricane_count = 1
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
                    hurricane_count += 1
                    new_hurricane = [{
                        'Date': row['Date'],
                        'Time': row['Time'],
                        'Lat': row['Latitude'],
                        'Long': row['Longitude'],
                        'MaxWind': row['Maximum Wind']
                    }]

            return hurricanes_dict

def main():
    predictor = HurricancePrediction()

    pacific_hurricanes = predictor.read('data/pacific.csv')
    atlantic_hurricanes = predictor.read('data/atlantic.csv')


if __name__ == "__main__":
    main()
