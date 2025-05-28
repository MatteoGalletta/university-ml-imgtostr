import math 

class ImageToStringPostprocessing:

    car_conf = {'c', 'j', 'k', 'o', 'p', 's', 'u', 'v', 'w', 'x', 'z',
            'C', 'J', 'K', 'O', 'P', 'S', 'U', 'V', 'W', 'X', 'Z'}
    car_non_conf_maiusc = {'A', 'B', 'D', 'E', 'F', 'G', 'H', 'I', 'L', 'M', 'N', 'Q', 'R', 'T', 'Y'}
    car_non_conf_minusc_b = {'g', 'm', 'y', 'o', 's', 'e', 'z', 'u', 'a', 'r', 'n', 'c', 'v', 'p', 'w', 'x', 'q'}
    car_non_conf_minusc_a = {'t', 'i', 'd', 'f', 'h', 'j', 'k', 'l', 'b'}

    def __init__(self):
        return
    
    def heuristics_spaces(self, info, labels):
        dist_dx = [x['dist_dx'] for x in info if x['dist_dx'] is not None]

        dist_dx_min = min(dist_dx)
        dist_dx_max = max(dist_dx)

        space_indexes = [i for i, v in enumerate(dist_dx) if v is not None and v > (dist_dx_min + dist_dx_max) * 0.6]

        # Insert spaces at the specified positions in the labels list
        labels_with_spaces = labels.copy()

        for idx in reversed(space_indexes):
            labels_with_spaces.insert(idx + 1, ' ')

        return "".join(labels_with_spaces)
    
    def heuristics_adjust(self, info):
        info_w_char_conf = [entry for entry in info if entry['char'] in self.car_conf]

        first_valid_entry = next(
            (entry for entry in info if entry['char'] not in self.car_conf),
            None
        )

        if first_valid_entry:
            print("Primo carattere valido trovato:", first_valid_entry['char'])
        else:
             print("Nessun carattere valido trovato.")

        delta = 5

        if first_valid_entry['char'] in self.car_non_conf_maiusc:
            for entry in info_w_char_conf:
                if math.isclose(first_valid_entry['dist_top'], entry['dist_top'], abs_tol=delta):
                    if entry['char'].islower():
                        entry['char'] = entry['char'].upper()
                else:
                    if entry['char'].isupper():
                        entry['char'] = entry['char'].lower()

        if first_valid_entry['char'] in self.car_non_conf_minusc_a:
            for entry in info_w_char_conf:
                if math.isclose(first_valid_entry['dist_top'], entry['dist_top'], abs_tol=delta):
                    if entry['char'].islower():
                        entry['char'] = entry['char'].upper()
                else:
                    if entry['char'].isupper():
                        entry['char'] = entry['char'].lower()

        if first_valid_entry['char'] in self.car_non_conf_minusc_b:
            for entry in info_w_char_conf:
                if math.isclose(first_valid_entry['dist_top'], entry['dist_top'], abs_tol=delta):
                    if entry['char'].isupper():
                        entry['char'] = entry['char'].lower()
                else:
                    if entry['char'].islower():
                        entry['char'] = entry['char'].upper()

        return info
