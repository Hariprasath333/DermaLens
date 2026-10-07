export interface AbcdeState {
  asymmetry: 0 | 1 | 2;
  border: 0 | 1 | 2;
  color: 0 | 1 | 2;
  diameter: 0 | 1;
  evolution: 0 | 1;
}

export const defaultAbcdeState: AbcdeState = {
  asymmetry: 0,
  border: 0,
  color: 0,
  diameter: 0,
  evolution: 0
};
