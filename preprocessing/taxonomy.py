import json
import os


class Taxonomy():
    UNRECOGNIZED_SUBTYPE_PREFIX = 'UNRECOGNIZED: '
    EXCLUDED_SUBTYPE_PREFIX = 'EXCLUDED: '

    def __init__(self):
        current_path = os.path.dirname(os.path.realpath(__file__))
        TAXONOMY_JSON = f'{current_path}/taxonomy.json'

        # Load JSON from a file
        with open(TAXONOMY_JSON, "r") as file:
            self._tree = json.load(file)

    # Recursive function to traverse the tree
    def _traverse_tree(nodes, level=0, exec_func=None):
        for node in nodes:
            if exec_func:
                exec_func(node, level)
            if "children" in node and node["children"]:
                Taxonomy._traverse_tree(node["children"], level + 1, exec_func)

    def print_tree(self, display_aliases=False):
        def print_node(node, level):
            aliases = ""
            if display_aliases and "aliases" in node:
                aliases = f" (Aliases: {', '.join(node['aliases'])})"
            print("  " * level + f"- {node['name']}{aliases}")
        Taxonomy._traverse_tree(self._tree, 0, print_node)

    # Recursive function to find the lineage of a node
    def _find_lineage(nodes, target_name, path=[]):
        for node in nodes:
            new_path = path + [node]  # Keep track of the lineage
            found = False
            if node["name"] == target_name:
                found = True
            elif "aliases" in node and target_name in node["aliases"]:
                found = True
            if found:
                return new_path
            if "children" in node and node["children"]:
                result = Taxonomy._find_lineage(
                    node["children"], target_name, new_path)
                if result:
                    return result
        return None  # Node not found

    def lookup(self,
               name,
               disease=None,
               join=False,
               remove_unrecognized_and_excluded=True,
               abbreviate=True):
        # Trim whitespace from name
        name = name.strip()

        def format_lineage_list(lineage_list, abbreviate=True):
            if abbreviate:
                disease = lineage_list[1]
                if len(lineage_list) < 3:
                    return disease
                return ','.join([disease, lineage_list[-1]])
            return ','.join(lineage_list)

        def format_unrecognized_and_excluded(prefix, disease=None):
            if remove_unrecognized_and_excluded:
                return None
            str = f'{prefix}{name}'
            if disease:
                return format_lineage_list([disease, str])
            return str

        if disease:
            parents = Taxonomy._find_lineage(self._tree, disease)
        else:
            parents = self._tree
        lineage = Taxonomy._find_lineage(parents, name)
        if not lineage:
            return format_unrecognized_and_excluded(
                Taxonomy.UNRECOGNIZED_SUBTYPE_PREFIX,
                disease)
        do_exclude = [node["exclude"] for node in lineage if "exclude" in node]
        if do_exclude:
            return format_unrecognized_and_excluded(
                Taxonomy.EXCLUDED_SUBTYPE_PREFIX,
                disease)
        result = [node["name"] for node in lineage]
        if join:
            return format_lineage_list(result, abbreviate)
        return result

    def translate_subtype_list(self,
                               subtypes,
                               disease=None,
                               join=False,
                               remove_unrecognized_and_excluded=False,
                               abbreviate=True,
                               split_delimiter=None):
        translated = {}
        for entry in subtypes:
            split_subtypes = entry.split(split_delimiter) if split_delimiter else [entry]
            if len(split_subtypes) > 1:
                translated_split_subtypes = []
                for x in split_subtypes:
                    translated_subtype = self.lookup(
                        x, disease, join, remove_unrecognized_and_excluded, abbreviate)
                    translated_split_subtypes.append(translated_subtype)
                translated[entry] = ','.join(translated_split_subtypes)
            else:
                translated[entry] = self.lookup(
                    entry, disease, join, remove_unrecognized_and_excluded, abbreviate)
        return translated

    def translate_subtype_column(self,
                                 column,
                                 disease=None,
                                 remove_unrecognized_and_excluded=False,
                                 abbreviate=False,
                                 split_delimiter=None):
        translated = self.translate_subtype_list(
            column,
            disease,
            join=True,
            remove_unrecognized_and_excluded=remove_unrecognized_and_excluded,
            abbreviate=abbreviate,
            split_delimiter=split_delimiter)
        return column.replace(translated)
