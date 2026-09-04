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
            elif "display_name" in node and node["display_name"] == target_name:
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
               exclude_subtypes=[],
               remove_unrecognized_and_excluded=True):
        name = name.strip()

        # If name is empty, return None
        if not name:
            return "NONE"

        def format_unrecognized_and_excluded(name, prefix, disease=None):
            if remove_unrecognized_and_excluded:
                return None
            return f'{prefix}{name}'

        lineage = Taxonomy._find_lineage(self._tree, name)
        if not lineage:
            return format_unrecognized_and_excluded(
                name,
                Taxonomy.UNRECOGNIZED_SUBTYPE_PREFIX)
        do_exclude = lineage[-1]["name"] in exclude_subtypes
        if do_exclude:
            return format_unrecognized_and_excluded(
                name,
                Taxonomy.EXCLUDED_SUBTYPE_PREFIX)
        return lineage[-1]["display_name"]

    def translate_subtype_list(self,
                               subtypes,
                               exclude_subtypes=[],
                               remove_unrecognized_and_excluded=False,
                               split_delimiter=None,
                               join_delimiter=';'):
        translated = {}
        for entry in subtypes:
            split_subtypes = entry.split(split_delimiter) if split_delimiter else [entry]
            if len(split_subtypes) > 1:
                translated_split_subtypes = []
                for x in split_subtypes:
                    translated_subtype = self.lookup(
                        x,
                        exclude_subtypes=exclude_subtypes,
                        remove_unrecognized_and_excluded=remove_unrecognized_and_excluded)
                    translated_split_subtypes.append(translated_subtype)
                translated[entry] = join_delimiter.join(translated_split_subtypes)
            else:
                translated[entry] = self.lookup(
                    entry,
                    exclude_subtypes=exclude_subtypes,
                    remove_unrecognized_and_excluded=remove_unrecognized_and_excluded)
        return translated

    def translate_subtype_column(self,
                                 column,
                                 exclude_subtypes=[],
                                 remove_unrecognized_and_excluded=False,
                                 split_delimiter=None,
                                 join_delimiter=';'):
        translated = self.translate_subtype_list(
            column,
            exclude_subtypes=exclude_subtypes,
            remove_unrecognized_and_excluded=remove_unrecognized_and_excluded,
            split_delimiter=split_delimiter,
            join_delimiter=join_delimiter
            )
        return column.replace(translated)
